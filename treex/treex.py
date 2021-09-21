import dataclasses
import enum
import functools
import inspect
import io
import threading
import typing as tp
from abc import ABCMeta
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import treeo as to
import yaml
from rich.console import Console
from rich.table import Table
from rich.text import Text

from treex import types, utils

A = tp.TypeVar("A")
B = tp.TypeVar("B")
T = tp.TypeVar("T", bound="ProtoModule")
Filter = tp.Union[
    tp.Type[tp.Type[tp.Any]],
    tp.Callable[[to.FieldInfo], bool],
]

PAD = r"{pad}"
LEAF_TYPES = (to.Nothing, types.Initializer, type(None))


@dataclass
class _Context(threading.local):
    call_info: tp.Optional[
        tp.Dict["ProtoModule", tp.Tuple[types.Inputs, tp.Any]]
    ] = None

    def __enter__(self):
        global _CONTEXT
        self._old_context = _CONTEXT
        _CONTEXT = self

    def __exit__(self, *args):
        global _CONTEXT
        _CONTEXT = self._old_context

    @contextmanager
    def update(self, **kwargs):
        fields = vars(self).copy()
        fields.pop("_old_context", None)
        fields.update(kwargs)

        with _Context(**fields):
            yield


_CONTEXT = _Context()

# -----------------------------------------
# ProtoModule
# -----------------------------------------


class ProtoModuleMeta(to.TreeMeta):
    def __call__(cls, *args, **kwargs) -> "ProtoModule":
        obj: ProtoModule = super().__call__(*args, **kwargs)

        if isinstance(obj, tp.Callable):
            new_call = _get_call(cls.__call__)
            cls.__call__ = functools.wraps(cls.__call__)(new_call)

        return obj


# override __call__
def _get_call(orig_call):
    def _meta_call(self: "ProtoModule", *args, **kwargs):
        outputs = orig_call(self, *args, **kwargs)

        if _CONTEXT.call_info is not None:
            inputs = types.Inputs(*args, **kwargs)
            _CONTEXT.call_info[self] = (inputs, outputs)

        return outputs

    return _meta_call


class ProtoModule(to.Tree, metaclass=ProtoModuleMeta):
    def module_init(self, key: jnp.ndarray) -> None:
        pass

    def __repr__(self) -> str:
        rep = _get_repr(self, level=0, array_type=None, inline=False)
        return _get_rich_repr(Text.from_markup(rep))

    def map(self: T, f: tp.Callable, *filters: Filter, inplace: bool = False) -> T:
        """
        Applies a function to all leaves in a pytree using `jax.tree_map`. If `filters` are given then
        the function will be applied only to the subset of leaves that match the filters.

        For example, if we want to zero all batch stats we can do:

        ```python
        module = SomeCompleModule()
        module = module.map(jnp.zeros_like, tx.BatchStat)
        ```

        Arguments:
            f: The function to apply to the leaves.
            filters: The filters used to select the leaves to which the function will be applied.
            inplace: If `True` then the object will be mutated with the changes.

        Returns:
            The object with the changes applied, if `inplace` is `True` then `self` is returned.
        """
        module = to.map(f, self, *filters)

        if inplace:
            self.__dict__.update(module.__dict__)
            return self
        else:
            return module

    def filter(self: T, *filters: Filter) -> T:
        """
        Creates a new module with the same structure, but sets to `Nothing` leaves that
        do not match **all** of the given filters. If a type `t` is given, the filter

        ```python
        _filter(x: Type[types.TreePart]) = issubclass(x, t)
        ```

        will be created for that type.

        Example:
        ```python
        class MyModule(tx.ProtoModule):
            a: tx.Parameter = 1
            b: tx.BatchStat = 2

        module = MyModule()

        module.filter(tx.Parameter) # MyModule(a=1, b=Nothing)
        module.filter(tx.BatchStat) # MyModule(a=Nothing, b=2)
        ```

        More fancy filters can be created by using callables:

        ```python
        # all States that are not OptStates
        module.filter(
            lambda field: issubclass(field.kind, tx.State)
            and not issubclass(field.kind, tx.OptState)
        )
        # MyModule(a=Nothing, b=2)
        ```

        Arguments:
            filters: Types to filter by, membership is determined by `issubclass`, or
                callables that take in a `FieldInfo` and return a `bool`.
        Returns:
            The new module with the filtered fields.

        """

        return to.filter(self, *filters)

    def update(self: T, other: T, *rest: T, inplace: bool = False) -> T:
        """
        Creates a new module with the same structure, but its values
        updated based on the values from the incoming modules. Updates are performed using
        the following rules:

        * Updates are performed in the order of the input modules from left to right.
        * If a leaf value from an incoming module is `Nothing`, it wont update
            the corresponding value on the currently aggregated module.
        * The static state of the output module (`initialized`, `training`, and user defined static fields)
            is the same as the current module (`self`).

        Example:

        ```python
        a = MyModule(x=Nothing, y=2, z=3)
        b = MyModule(x=1, y=Nothing, z=4)

        a.update(b) # MyModule(x=1, y=2, z=4)
        ```
        Notice the following:

        * The value of `x` and `z` were updated since they were present in `b`.
        * The value of `y` was not updated since `b.y` was `Nothing`.

        When using `update` with multiple modules the following equivalence holds:

        ```
        m1.update(m2, m3) = m1.update(m2).update(m3)
        ```

        If you want to update the current module instead of creating a new one use `inplace=True`.
        This is useful when applying transformation inside a method where reassigning `self` is not possible:

        ```python
        def double_params(self):
            # this is not doing what you expect
            self = jax.tree_map(lambda x: 2 * x, self)
        ```
        Instead do this:

        ```python
        def double_params(self):
            doubled = jax.tree_map(lambda x: 2 * x, self)
            self.update(doubled, inplace=True)
        ```

        Arguments:
            other: The first to get the values to update from.
            rest: Additional modules to perform the update in order from left to right.
            inplace: If `True`, the current module is modified with the updated values.

        Returns:
            A new module with the updated values or `None` if `inplace` is `True`.
        """
        module = to.update(self, other, *rest)

        if inplace:
            self.__dict__.update(module.__dict__)
            return self
        else:
            return module

    def tabulate(
        self,
        inputs: tp.Union[tp.Any, types.Inputs, None] = None,
        depth: int = -1,
        signature: bool = False,
        param_types: bool = True,
    ) -> str:
        """
        Returns a tabular representation of the module.

        Arguments:
            depth: The maximum depth of the representation in terms of nested Modules, -1 means no limit.
            signature: Whether to show the signature of the ProtoModule.
            param_types: Whether to show the types of the parameters.
        Returns:
            A string containing the tabular representation.
        """
        self = self.copy()

        if inputs is not None:
            if not isinstance(inputs, types.Inputs):
                inputs = types.Inputs(inputs)

            inputs = tp.cast(types.Inputs, inputs)

            if not isinstance(self, tp.Callable):
                raise TypeError(
                    "`inputs` can only be specified if the module is a callable."
                )

            with _Context(call_info={}):

                # call using self to preserve references
                def eval_call(args, kwargs):
                    assert isinstance(self, tp.Callable)
                    return self(*args, **kwargs)

                jax.eval_shape(
                    eval_call,
                    inputs.args,
                    inputs.kwargs,
                )
                call_info = _CONTEXT.call_info

        else:
            call_info = None

        with to.add_field_info():
            flat: tp.List[to.FieldInfo]
            flat, _ = jax.tree_flatten(self)
            tree_part_types: tp.Tuple[tp.Type[types.TreePart], ...] = tuple(
                {
                    field_info.kind
                    for field_info in flat
                    if _generic_issubclass(field_info.kind, types.TreePart)
                }
            )

        path = ()
        rows = list(
            _get_tabulate_rows(
                path, self, depth, tree_part_types, signature, param_types
            )
        )

        modules = [row[0] for row in rows]
        rows = [row[1:] for row in rows]

        if call_info is not None:
            for module, row in zip(modules, rows):
                if module in call_info:
                    inputs, outputs = call_info[module]
                    simplified_inputs = (
                        inputs.args[0]
                        if len(inputs.kwargs) == 0 and len(inputs.args) == 1
                        else inputs.kwargs
                        if len(inputs.kwargs) == 0
                        else inputs.kwargs
                        if len(inputs.args) == 0
                        else (inputs.args, inputs.kwargs)
                    )

                    inputs_repr = _format_param_tree(simplified_inputs)
                    outputs_repr = _format_param_tree(outputs)
                else:
                    inputs_repr = ""
                    outputs_repr = ""

                row.insert(3, outputs_repr)
                row.insert(3, inputs_repr)

        n_non_treepart_cols = 2 if call_info is None else 4

        rows[0][0] = "*"
        rows.append(
            [""] * n_non_treepart_cols
            + ["Total:"]
            + [
                _format_obj_size(self.filter(kind), add_padding=True)
                for kind in tree_part_types
            ]
        )
        _add_padding(rows)

        table = Table(
            show_header=True,
            show_lines=True,
            show_footer=True,
            # box=rich.box.HORIZONTALS,
        )

        table.add_column("path")
        table.add_column("module")
        table.add_column("params")

        if call_info is not None:
            table.add_column("inputs")
            table.add_column("outputs")

        for tree_part_type in tree_part_types:
            type_name = tree_part_type.__name__
            if type_name.startswith("_"):
                type_name = type_name[1:]

            table.add_column(type_name)

        for row in rows[:-1]:
            table.add_row(*row)

        table.columns[n_non_treepart_cols].footer = Text.from_markup(
            rows[-1][n_non_treepart_cols], justify="right"
        )

        for i in range(len(tree_part_types)):
            table.columns[n_non_treepart_cols + 1 + i].footer = rows[-1][
                n_non_treepart_cols + 1 + i
            ]

        table.caption_style = "bold"
        table.caption = "\nTotal Parameters: " + _format_obj_size(
            self, add_padding=False
        )

        return _get_rich_repr(table)

    # --------------------------------------
    # filter shortcuts
    # --------------------------------------

    def parameters(self: T, *filters: Filter) -> T:
        """
        Returns a copy of the ProtoModule with only tx.Parameter TreeParts, alias for `filter(tx.Parameter)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.Parameter, *filters)

    def batch_stats(self: T, *filters: Filter) -> T:
        """
        Returns a copy of the ProtoModule with only tx.BatchStat TreeParts, alias for `filter(tx.BatchStat)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.BatchStat, *filters)

    def rngs(self: T, *filters: Filter) -> T:
        """
        Returns a copy of the ProtoModule with only tx.Rng TreeParts, alias for `filter(tx.Rng)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.Rng, *filters)

    def model_states(self: T, *filters: Filter) -> T:
        """
        Returns a copy of the ProtoModule with only tx.ModelState TreeParts, alias for `filter(tx.ModelState)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.ModelState, *filters)

    def states(self: T, *filters: Filter) -> T:
        """
        Returns a copy of the ProtoModule with only tx.State TreeParts, alias for `filter(tx.State)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.State, *filters)

    def metrics(self: T, *filters: Filter) -> T:
        """
        Returns a copy of the ProtoModule with only tx.Metric TreeParts, alias for `filter(tx.Metric)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.Metric, *filters)

    def losses(self: T, *filters: Filter) -> T:
        """
        Returns a copy of the ProtoModule with only tx.Loss TreeParts, alias for `filter(tx.Loss)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.Loss, *filters)

    def logs(self: T, *filters: Filter) -> T:
        """
        Returns a copy of the ProtoModule with only tx.Log TreeParts, alias for `filter(tx.Log)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.Log, *filters)


# --------------------------------------------------
# utils
# --------------------------------------------------


def _get_rich_repr(table):
    f = io.StringIO()
    console = Console(file=f, force_terminal=True)
    console.print(table)

    return f.getvalue()


def _get_repr(
    obj, level: int, array_type: tp.Optional[type], inline, space="  "
) -> str:
    indent_level = space * level

    if isinstance(obj, ProtoModule):

        (tree,), _ = obj.tree_flatten()

        params = {}
        submodules = {}

        for field, value in tree.items():
            annotation: tp.Union[
                tp.Type[ProtoModule], tp.Type[types.TreePart], None
            ] = _first_issubclass(
                obj.field_metadata[field].kind, (types.TreePart, ProtoModule)
            )

            if annotation is None:
                continue
            if _generic_issubclass(annotation, ProtoModule):
                submodules[field] = value
            elif _generic_issubclass(annotation, types.TreePart):
                params[field] = value
            else:
                continue

        body = [
            indent_level
            + space
            + f"{field}: {_get_repr(value, level + 1, obj.field_metadata[field].kind, inline=True)}"
            for field, value in params.items()
        ] + [
            indent_level
            + space
            + f"{field}: {_get_repr(value, level + 1, obj.field_metadata[field].kind, inline=True)}"
            for field, value in submodules.items()
        ]

        body_str = "\n".join(body)
        end_dot = ":" if not inline else ""
        type_str = (
            f"[dim]{obj.__class__.__name__}[/]" if inline else obj.__class__.__name__
        )
        # signature_repr = _format_module_signature(obj, not_tree)

        return f"{type_str}{end_dot}\n{body_str}"

    elif isinstance(obj, tp.Mapping):
        body = [
            indent_level
            + space
            + f"{field}: {_get_repr(value, level + 1, array_type, inline=True)}"
            for field, value in obj.items()
        ]
        body_str = "\n".join(body)
        end_dot = ":" if not inline else ""
        type_str = (
            f"[dim]{obj.__class__.__name__}[/]" if inline else obj.__class__.__name__
        )
        return f"{type_str}{end_dot}\n{body_str}"
        # return f"\n{body_str}"
    elif isinstance(obj, tp.Sequence):
        body = [
            indent_level
            + space
            + f"- {_get_repr(value, level + 2, array_type, inline=False)}"
            for i, value in enumerate(obj)
        ]
        body_str = "\n".join(body)
        end_dot = ":" if not inline else ""
        type_str = (
            f"[dim]{obj.__class__.__name__}[/]" if inline else obj.__class__.__name__
        )
        return f"{type_str}{end_dot}\n{body_str}"

    elif isinstance(obj, (np.ndarray, jnp.ndarray)):
        assert array_type is not None

        type_name = array_type.__name__
        if type_name.startswith("_"):
            type_name = type_name[1:]

        shape = ", ".join(str(x) for x in obj.shape)
        return f"{type_name}([green]{shape}[/]) [dim]{obj.dtype}[/]"
    else:
        return repr(obj)


def _get_tabulate_rows(
    path,
    obj,
    level,
    tree_part_types,
    include_signature,
    include_param_type,
) -> tp.Iterable[tp.List[tp.Any]]:
    if isinstance(obj, ProtoModule):
        (tree,), not_tree = obj.tree_flatten()

        params = {}
        submodules = {}

        for field, value in tree.items():
            annotation: tp.Union[
                tp.Type[ProtoModule], tp.Type[types.TreePart], None
            ] = _first_issubclass(
                obj.field_metadata[field].kind, (types.TreePart, ProtoModule)
            )

            if annotation is None:
                continue
            if issubclass(annotation, ProtoModule):
                submodules[field] = value
            elif issubclass(annotation, types.TreePart):
                params[field] = value
            else:
                continue

        path_str = "".join(str(x) for x in path)
        signature = _format_module_signature(obj) if include_signature else {}
        argumentes_str = ", ".join(
            f"{arg} = {value}" for arg, value in signature.items()
        )
        signature_str = f"{obj.__class__.__name__}({argumentes_str})"

        if len(signature_str) > 40:
            argumentes_str = ",\n  ".join(
                f"{arg} = {value}" for arg, value in signature.items()
            )
            signature_str = f"{obj.__class__.__name__}(\n  {argumentes_str}\n)"

        params_repr = {
            field: jax.tree_map(
                lambda x: str(x)
                if isinstance(x, LEAF_TYPES)
                else _format_param(
                    x, obj.field_metadata[field].kind, include_param_type
                ),
                value,
                is_leaf=lambda x: isinstance(x, LEAF_TYPES),
            )
            for field, value in params.items()
        }
        params_repr = _simplify(params_repr)
        params_str = _as_yaml_str(params_repr)

        if level == 0:
            tree_part_sizes = [
                _format_obj_size(obj.filter(t), add_padding=True)
                for t in tree_part_types
            ]
        else:
            tree_part_sizes = [
                _format_obj_size(
                    [
                        value
                        for field, value in params.items()
                        if obj.field_metadata[field].kind is t
                    ],
                    add_padding=True,
                )
                for t in tree_part_types
            ]

        yield [obj, path_str, signature_str, params_str] + tree_part_sizes

        if level != 0:
            for field, value in submodules.items():
                yield from _get_tabulate_rows(
                    path + (f".{field}",),
                    value,
                    level - 1,
                    tree_part_types,
                    include_signature,
                    include_param_type,
                )
    elif isinstance(obj, tp.Mapping):
        for field, value in obj.items():
            if isinstance(field, str):
                field = f'"{field}"'

            yield from _get_tabulate_rows(
                path + (f"[{field}]",),
                value,
                level,
                tree_part_types,
                include_signature,
                include_param_type,
            )
    elif isinstance(obj, tp.Sequence):
        for i, value in enumerate(obj):
            yield from _get_tabulate_rows(
                path + (f"[{i}]",),
                value,
                level,
                tree_part_types,
                include_signature,
                include_param_type,
            )
    else:
        raise ValueError(f"Unsupported type {type(obj)}")


def _as_yaml_str(value) -> str:
    if hasattr(value, "__len__") and len(value) == 0:
        return ""

    file = io.StringIO()
    yaml.safe_dump(
        value,
        file,
        default_flow_style=False,
        indent=2,
        sort_keys=False,
        explicit_end=False,
    )
    return file.getvalue().replace("\n...", "").replace("'", "")


def _simplify(obj):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, tp.Sequence):
        return [_simplify(x) for x in obj]
    elif isinstance(obj, tp.Mapping):
        return {k: _simplify(v) for k, v in obj.items()}
    else:
        return obj


def _format_module_signature(obj: ProtoModule) -> tp.Dict[str, str]:
    signature = {}
    simple_types = {int, str, float, bool}
    cls = obj.__class__
    values = list(inspect.signature(cls.__init__).parameters.values())[1:]
    names = {x.name for x in values}

    for field, value in vars(obj).items():
        if field in names:
            all_types = set(jax.tree_leaves(jax.tree_map(type, value)))

            if all_types.issubset(simple_types):
                signature[field] = value
            elif isinstance(value, to.ArrayLike) and value.shape == ():
                signature[field] = value

    signature = jax.tree_map(
        lambda x: f'"{x}"'
        if isinstance(x, str)
        else (
            f"{x:.2e}"
            if x < 1e-2
            else f"{x:.3f}"
            if x < 1
            else f"{x:.2f}"
            if x < 100
            else f"{x:.1f}"
            if x < 1000
            else f"{x:.2e}"
        )
        if isinstance(x, float)
        else str(x),
        signature,
    )
    signature = {
        field: str(value).replace("'", "") for field, value in signature.items()
    }

    return signature


def _format_param(obj, array_type, include_param_type) -> str:

    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        if include_param_type:
            type_name = array_type.__name__
            if type_name.startswith("_"):
                type_name = type_name[1:]
        else:
            type_name = ""

        shape = ", ".join(str(x) for x in obj.shape)
        return f"{type_name}([green]{shape}[/]){PAD}  [dim]{obj.dtype}[/]"
    else:
        return str(obj) if include_param_type else f"() [dim]{type(obj).__name__}[/]"


def _format_param_tree(params) -> str:

    params_repr = jax.tree_map(
        lambda x: _format_param(x, None, include_param_type=False),
        params,
        is_leaf=lambda x: isinstance(x, to.Nothing),
    )

    params_repr = _simplify(params_repr)
    params_str = _as_yaml_str(params_repr)

    return params_str


def _format_obj_size(obj, add_padding):
    count = sum((x.size if hasattr(x, "size") else 0 for x in jax.tree_leaves(obj)), 0)
    size = sum(
        (x.nbytes if hasattr(x, "nbytes") else 0 for x in jax.tree_leaves(obj)), 0
    )
    padding = f"{PAD}" if add_padding else ""

    return (
        f"[green]{count:,}[/]{padding}  [dim]{_format_size(size)}[/]"
        if count > 0
        else ""
    )


def _add_padding(rows):
    n_cols = len(rows[0])

    for col in range(n_cols):
        max_length = max(
            len(line.split(PAD)[0]) for row in rows for line in row[col].split("\n")
        )

        for row in rows:
            if PAD in row[col]:
                row[col] = "\n".join(
                    line.format(
                        pad=" " * (max_length - len(line.rstrip().split(PAD)[0]))
                    )
                    if PAD in line
                    else line
                    for line in row[col].rstrip().split("\n")
                )


def _format_size(size):
    count, units = (
        (f"{size / 1e9 :,.1f}", "GB")
        if size > 1e9
        else (f"{size / 1e6 :,.1f}", "MB")
        if size > 1e6
        else (f"{size / 1e3 :,.1f}", "KB")
        if size > 1e3
        else (f"{size:,}", "B")
    )

    return f"{count}{units}"


def _safe_issubclass(a, b) -> bool:
    return issubclass(a, b) if isinstance(a, tp.Type) else issubclass(type(a), b)


def _generic_issubclass(__cls, __class_or_tuple) -> bool:
    return _safe_issubclass(__cls, __class_or_tuple) or (
        hasattr(__cls, "__origin__")
        and _safe_issubclass(__cls.__origin__, __class_or_tuple)
    )


def _first_issubclass(cls: type, __class_or_tuple) -> tp.Optional[tp.Type[tp.Any]]:
    for t in to.utils._all_types(cls):
        if issubclass(t, __class_or_tuple):
            return t


def _remove_generic(_cls):
    return (
        _cls
        if isinstance(_cls, type)
        else _cls.__origin__
        if hasattr(_cls, "__origin__")
        else type(_cls)
    )
