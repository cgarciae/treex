import inspect
import io
import threading
import typing as tp
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from numpy.core.fromnumeric import shape
from rich.console import Console
import yaml
from rich.table import Table
from rich.text import Text

from treex import types

A = tp.TypeVar("A")
B = tp.TypeVar("B")
T = tp.TypeVar("T", bound="TreeObject")

PAD = r"{pad}"


class Context(threading.local):
    is_slicing: bool = False
    is_initializing: bool = False
    training: tp.Optional[bool] = None
    key: tp.Optional[jnp.ndarray] = None

    def next_key(self) -> jnp.ndarray:
        assert self.is_initializing and self.key is not None

        # key = self.key
        key, self.key = jax.random.split(self.key)

        return key


LOCAL: Context = Context()


class TreeObject:
    def _get_props(self) -> tp.Dict[str, tp.Any]:
        return {}

    def tree_flatten(self):
        annotations = getattr(self.__class__, "__annotations__", {})
        fields = vars(self)

        tree = {}
        not_tree = {}

        for name, value in fields.items():
            annotation = annotations.get(name, None)
            annotation = _resolve_tree_type(name, annotation)

            if annotation is None:
                not_tree[name] = value

            elif issubclass(annotation, TreeObject):
                tree[name] = value

            elif issubclass(annotation, types.TreePart):
                if LOCAL.is_slicing:
                    tree[name] = jax.tree_map(
                        lambda x: types.ValueAnnotation(x, annotation), value
                    )
                else:
                    tree[name] = value
            else:
                not_tree[name] = value

        return tuple(tree.values()), dict(
            tree=tree.keys(),
            not_tree=not_tree,
            props=self._get_props(),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        module = cls.__new__(cls)

        for i, k in enumerate(aux_data["tree"]):
            setattr(module, k, children[i])

        for k, v in aux_data["not_tree"].items():
            setattr(module, k, v)

        for k, v in aux_data["props"].items():
            setattr(module, k, v)

        if LOCAL.is_initializing and not module._initialized:
            module.module_init(LOCAL.next_key())
            module._initialized = True

        if LOCAL.training is not None:
            module._training = LOCAL.training

        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

    def copy(self: T) -> T:
        """
        Returns a deep copy of the module, implemented as:
        ```python
        jax.tree_map(lambda x: x, self)
        ```
        """
        return jax.tree_map(lambda x: x, self)

    def __repr__(self) -> str:
        rep = _get_repr(self, level=0, array_type=None, inline=False)
        return _get_rich_repr(Text.from_markup(rep))


class Module(TreeObject):
    _initialized = False
    _training = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def training(self) -> bool:
        return self._training

    def _get_props(self) -> tp.Dict[str, tp.Any]:
        return dict(_initialized=self._initialized, _training=self._training)

    def init(self: T, key: tp.Union[int, jnp.ndarray]) -> T:
        """
        Creates a new module with the same structure, but with its fields initialized given a seed `key`. The following
        procedure is used:

        1. The input `key` is split and iteratively updated before passing a derived value to any
            process that requires initialization.
        2. `Initializer`s are called and applied to the module first.
        3. `TreeObject.module_init` methods are called last.

        Arguments:
            key: The seed to use for initialization.
        Returns:
            The new module with the fields initialized.
        """
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)

        old_initializing = LOCAL.is_initializing
        old_key = LOCAL.key

        LOCAL.is_initializing = True
        LOCAL.key = key

        try:
            module = jax.tree_map(
                lambda initializer: (
                    initializer(LOCAL.next_key())
                    if isinstance(initializer, types.Initializer)
                    else initializer
                ),
                self,
            )
        finally:
            LOCAL.is_initializing = old_initializing
            LOCAL.key = old_key

        return module

    def module_init(self, key: jnp.ndarray) -> None:
        pass

    def filter(self: T, *filters: tp.Type) -> T:
        """
        Creates a new module with the same structure, but leaves whose type annotations are not subtypes
        of the given filters (as determined by `issubclass`) as set to `Nothing`.

        Example:
        ```python
        class MyModule(tx.TreeObject):
            a: tx.Parameter = 1
            b: tx.BatchStat = 2

        module = MyModule()

        module.filter(tx.Parameter) # MyModule(a=1, b=Nothing)
        module.filter(tx.BatchStat) # MyModule(a=Nothing, b=2)
        ```

        Arguments:
            filters: Types to filter by, membership is determined by `issubclass`.
        Returns:
            The new module with the filtered fields.

        """
        flat: tp.List[types.ValueAnnotation]

        old_slicing = LOCAL.is_slicing
        LOCAL.is_slicing = True

        try:
            flat, treedef = jax.tree_flatten(self)
            flat_out = [
                value_annotation.value
                if issubclass(value_annotation.annotation, filters)
                else types.Nothing()
                for value_annotation in flat
            ]
            module = jax.tree_unflatten(treedef, flat_out)
        finally:
            LOCAL.is_slicing = old_slicing

        return module

    @tp.overload
    def update(self: T, other: T, *rest: T) -> T:
        ...

    @tp.overload
    def update(self: T, other: T, *rest: T, inplace: bool) -> None:
        ...

    def update(self: T, other: T, *rest: T, inplace: bool = False) -> tp.Optional[T]:
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
        modules = (self, other) + rest

        def merge_fn(xs):
            acc, *xs = xs
            for x in xs:
                if not isinstance(x, types.Nothing):
                    acc = x
            return acc

        flats, treedefs = zip(
            *[
                jax.tree_flatten(m, is_leaf=lambda x: isinstance(x, types.Nothing))
                for m in modules
            ]
        )
        # flat_out = jax.tree_util.tree_map(merge_fn, *flats)
        flat_out = [merge_fn(values) for values in zip(*flats)]
        module = jax.tree_unflatten(treedefs[0], flat_out)

        if inplace:
            self.__dict__.update(module.__dict__)
            return None
        else:
            return module

    def train(self: T, mode: bool = True) -> T:
        """
        Creates a new module with the same structure, but with `TreeObject.training` set to the given value.

        Arguments:
            mode: The new training mode.
        Returns:
            The new module in with the training mode is set to the given value.
        """
        old_training = LOCAL.training
        LOCAL.training = mode

        try:
            module = self.copy()  # trigger flatten / unflatten
        finally:
            LOCAL.training = old_training

        return module

    def eval(self: T) -> T:
        """
        Creates a new module with the training mode set to False, equivalent to calling `train(False)`.

        Returns:
            The new module with the training mode set to False.
        """
        return self.train(False)

    def tabulate(
        self, depth: int = -1, signature: bool = False, param_types: bool = True
    ) -> str:
        """
        Returns a tabular representation of the module.

        Arguments:
            depth: The maximum depth of the representation in terms of nested Modules, -1 means no limit.
            signature: Whether to show the signature of the TreeObject.
            param_types: Whether to show the types of the parameters.
        Returns:
            A string containing the tabular representation.
        """
        old_slicing = LOCAL.is_slicing
        LOCAL.is_slicing = True

        try:
            flat, _ = jax.tree_flatten(self)
            tree_part_types: tp.Tuple[tp.Type[types.TreePart], ...] = tuple(
                {value_annotation.annotation for value_annotation in flat}
            )
        finally:
            LOCAL.is_slicing = old_slicing

        path = ()
        rows = list(
            _get_tabulate_rows(
                path, self, depth, tree_part_types, signature, param_types
            )
        )
        rows[0][0] = "*"
        rows.append(
            ["", "", "Total:"]
            + [
                _format_obj_size(self.filter(tree_type), add_padding=True)
                for tree_type in tree_part_types
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

        for tree_part_type in tree_part_types:
            type_name = tree_part_type.__name__
            if type_name.startswith("_"):
                type_name = type_name[1:]

            table.add_column(type_name)

        for row in rows[:-1]:
            table.add_row(*row)

        table.columns[2].footer = Text.from_markup(rows[-1][2], justify="right")

        for i in range(len(tree_part_types)):
            table.columns[3 + i].footer = rows[-1][3 + i]

        table.caption_style = "bold"
        table.caption = "\nTotal Parameters: " + _format_obj_size(
            self, add_padding=False
        )

        return _get_rich_repr(table)


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

    if isinstance(obj, TreeObject):

        annotations = getattr(obj.__class__, "__annotations__", {})
        children, aux_data = obj.tree_flatten()

        params = {}
        submodules = {}

        for i, field in enumerate(aux_data["tree"]):
            annotation = annotations.get(field, None)
            annotation = _resolve_tree_type(field, annotation)
            annotations[field] = annotation

            assert annotation is not None

            if issubclass(annotation, TreeObject):
                submodules[field] = children[i]
            elif issubclass(annotation, types.TreePart):
                params[field] = children[i]
            else:
                raise ValueError(f"Unsupported type {annotation}")

        body = [
            indent_level
            + space
            + f"{field}: {_get_repr(value, level + 1, annotations[field], inline=True)}"
            for field, value in params.items()
        ] + [
            indent_level
            + space
            + f"{field}: {_get_repr(value, level + 1, annotations[field], inline=True)}"
            for field, value in submodules.items()
        ]

        body_str = "\n".join(body)
        end_dot = ":" if not inline else ""
        type_str = (
            f"[dim]{obj.__class__.__name__}[/]" if inline else obj.__class__.__name__
        )
        # signature_repr = _format_module_signature(obj, aux_data["not_tree"])

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
    if isinstance(obj, TreeObject):
        annotations = getattr(obj.__class__, "__annotations__", {})
        children, aux_data = obj.tree_flatten()

        params = {}
        submodules = {}

        for i, field in enumerate(aux_data["tree"]):
            annotation = annotations.get(field, None)
            annotation = _resolve_tree_type(field, annotation)
            annotations[field] = annotation

            assert annotation is not None

            if issubclass(annotation, TreeObject):
                submodules[field] = children[i]
            elif issubclass(annotation, types.TreePart):
                params[field] = children[i]
            else:
                raise ValueError(f"Unsupported type {annotation}")

        path_str = "".join(str(x) for x in path)
        signature = (
            _format_module_signature(obj, aux_data["not_tree"])
            if include_signature
            else {}
        )
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
                lambda x: _format_param(x, annotations[field], include_param_type),
                value,
                is_leaf=lambda x: isinstance(x, types.Nothing),
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
                        if annotations[field] is t
                    ],
                    add_padding=True,
                )
                for t in tree_part_types
            ]

        yield [path_str, signature_str, params_str] + tree_part_sizes

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


def _format_module_signature(
    obj: TreeObject, not_tree: tp.Dict[str, tp.Any]
) -> tp.Dict[str, str]:
    signature = {}
    simple_types = {int, str, float, bool}
    cls = obj.__class__
    values = list(inspect.signature(cls.__init__).parameters.values())[1:]

    for field in values:
        if field.name in not_tree:
            value = not_tree[field.name]
            types = set(jax.tree_leaves(jax.tree_map(type, value)))

            if types.issubset(simple_types):
                signature[field.name] = value

    signature = jax.tree_map(
        lambda x: f'"{x}"'
        if isinstance(x, str)
        else f"{x:.4f}"
        if isinstance(x, float)
        else str(x),
        signature,
    )

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


def _resolve_tree_type(name: str, t: tp.Optional[type]) -> tp.Optional[type]:
    if t is None:
        return None

    tree_types = [
        x
        for x in _all_types(t)
        if isinstance(x, tp.Type)
        if issubclass(x, (types.TreePart, TreeObject))
    ]

    if len(tree_types) > 1:
        # if its a type with many TreeObject subtypes just mark them all as TreeObject
        if all(issubclass(x, TreeObject) for x in tree_types):
            return TreeObject
        else:
            raise TypeError(
                f"Multiple tree parts found in annotation for field '{name}': {tree_types}"
            )
    elif len(tree_types) == 1:
        return tree_types[0]
    else:
        return None


def _all_types(t: tp.Type) -> tp.Iterable[tp.Type]:
    if hasattr(t, "__args__"):
        for arg in t.__args__:
            yield from _all_types(arg)
    else:
        yield t
