import functools
import threading
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import treeo as to
from rich import inspect
from rich.table import Table
from rich.text import Text
from treeo.utils import _get_unbound_method

from treex import types, utils
from treex.treex import Filters, Treex

A = tp.TypeVar("A")
B = tp.TypeVar("B")
C = tp.TypeVar("C", bound=tp.Callable[..., tp.Any])
M = tp.TypeVar("M", bound="Module")


@dataclass
class _ModuleContext(threading.local):
    key: tp.Optional[jnp.ndarray] = None
    initializing: bool = False

    def __enter__(self):
        global _MODULE_CONTEXT
        self._old_context = _MODULE_CONTEXT
        _MODULE_CONTEXT = self

    def __exit__(self, *args):
        global _MODULE_CONTEXT
        _MODULE_CONTEXT = self._old_context

    @contextmanager
    def update(self, **kwargs):
        fields = vars(self).copy()
        fields.pop("_old_context", None)
        fields.update(kwargs)

        with _ModuleContext(**fields):
            yield


@dataclass
class _CallContext(threading.local):
    call_info: tp.Optional[tp.Dict["Module", tp.Tuple[types.Inputs, tp.Any]]] = None

    def __enter__(self):
        global _CALL_CONTEXT
        self._old_context = _CALL_CONTEXT
        _CALL_CONTEXT = self

    def __exit__(self, *args):
        global _CALL_CONTEXT
        _CALL_CONTEXT = self._old_context

    @contextmanager
    def update(self, **kwargs):
        fields = vars(self).copy()
        fields.pop("_old_context", None)
        fields.update(kwargs)

        with _CallContext(**fields):
            yield


_CALL_CONTEXT = _CallContext()
_MODULE_CONTEXT = _ModuleContext()

# -----------------------------------------
# Module
# -----------------------------------------
class ModuleMeta(to.TreeMeta):
    def construct(cls, obj: M, *args, **kwargs) -> M:
        # reset context during construction
        with _ModuleContext():
            obj = super().construct(obj, *args, **kwargs)

        if not hasattr(obj, "name"):
            obj.name = utils._lower_snake_case(obj.__class__.__name__)

        if to.in_compact():
            if _MODULE_CONTEXT.key is None:
                raise RuntimeError(
                    f"Trying to construct new module {obj} with a compact context outside of `init` or an `rng_key` context."
                )

            def call_rng_init(module: Module):
                if isinstance(module, Module) and not module._initialized:
                    module.setup()

            obj = to.apply(call_rng_init, obj)

        return obj


class Module(Treex, Filters, metaclass=ModuleMeta):
    # use to.field to copy class vars to instance
    _training: bool = to.static(True)
    _initialized: bool = to.static(False)
    _frozen: bool = to.static(False)

    def __init__(self, name: tp.Optional[str] = None):
        self.name = (
            name
            if name is not None
            else utils._lower_snake_case(self.__class__.__name__)
        )

    def initializing(self) -> bool:
        if not self.initialized:
            if not _MODULE_CONTEXT.initializing:
                raise RuntimeError(
                    f"Trying run {self.__class__.__name__} for the first time outside of `init`"
                )

            return True

        return False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def training(self) -> bool:
        return self._training

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __init_subclass__(cls):
        if issubclass(cls, tp.Callable):
            orig_call = cls.__call__

            @functools.wraps(cls.__call__)
            def new_call(self: Module, *args, **kwargs):
                outputs = orig_call(self, *args, **kwargs)

                if (
                    _CALL_CONTEXT.call_info is not None
                    and self not in _CALL_CONTEXT.call_info
                ):
                    inputs = types.Inputs(*args, **kwargs)
                    _CALL_CONTEXT.call_info[self] = (inputs, outputs)

                return outputs

            cls.__call__ = new_call

        return super().__init_subclass__()

    @tp.overload
    def init(
        self: M,
        *,
        key: tp.Optional[tp.Union[int, jnp.ndarray]],
        method: tp.Union[str, tp.Callable] = "__call__",
    ) -> tp.Callable[..., M]:
        ...

    @tp.overload
    def init(
        self: M,
        *,
        return_output: tp.Literal[True],
        key: tp.Optional[tp.Union[int, jnp.ndarray]],
        method: tp.Union[str, tp.Callable] = "__call__",
    ) -> tp.Callable[..., tp.Union[M, tp.Tuple[tp.Any, M]]]:
        ...

    @tp.overload
    def init(
        self: M,
        *,
        return_output: tp.Literal[False],
        key: tp.Optional[tp.Union[int, jnp.ndarray]],
        method: tp.Union[str, tp.Callable] = "__call__",
    ) -> tp.Callable[..., M]:
        ...

    def init(
        self: M,
        *,
        key: tp.Optional[tp.Union[int, jnp.ndarray]],
        method: tp.Union[str, tp.Callable] = "__call__",
        return_output: bool = False,
    ) -> tp.Callable[..., tp.Union[M, tp.Tuple[tp.Any, M]]]:
        """
        Method version of `tx.init`, it applies `self` as first argument.

        `init` creates a new module with the same structure, but with its fields initialized given a seed `key`. The following
        procedure is used:

        1. The input `key` is split and iteratively updated before passing a derived value to any
            process that requires initialization.
        2. `Module.setup` methods are called last.

        Arguments:
            key: The seed to use for initialization.
        Returns:
            The new module with the fields initialized.
        """
        key = utils.Key(key) if key is not None else None

        def init_fn(*args, **kwargs) -> tp.Union[M, tp.Tuple[tp.Any, M]]:
            module: M = self

            with _MODULE_CONTEXT.update(key=key, initializing=True):

                def call_rng_init(module: Module):
                    if isinstance(module, Module) and not module._initialized:
                        module.setup()

                module = to.apply(call_rng_init, module)

                output, module = module.apply(
                    key=_MODULE_CONTEXT.key,
                    method=method,
                    mutable=True,
                )(*args, **kwargs)

                def set_initialized(module: Module):
                    if isinstance(module, Module) and not module._initialized:
                        module._initialized = True

                module = to.apply(set_initialized, module)

            if return_output:
                return output, module
            else:
                return module

        return init_fn

    def apply(
        self: M,
        *,
        key: tp.Optional[types.KeyLike] = None,
        method: tp.Union[str, tp.Callable] = "__call__",
        mutable: bool = True,
    ) -> tp.Callable[..., tp.Tuple[tp.Any, M]]:

        key = utils.Key(key) if key is not None else None
        unbounded_method = _get_unbound_method(self, method)

        def apply_fn(*args, **kwargs) -> tp.Tuple[tp.Any, M]:
            with _MODULE_CONTEXT.update(key=key):

                if mutable:
                    return to.mutable(unbounded_method)(self, *args, **kwargs)
                else:
                    module = self.copy()
                    output = unbounded_method(module, *args, **kwargs)
                    return output, module

        return apply_fn

    def setup(self) -> None:
        pass

    @staticmethod
    def next_key() -> jnp.ndarray:
        return next_key()

    def tabulate(
        self,
        *args,
        summary_depth: int = -1,
        show_signatures: bool = False,
        show_param_types: bool = True,
        **kwargs,
    ) -> str:
        """
        Returns a tabular representation of the module.

        Arguments:
            depth: The maximum depth of the representation in terms of nested Modules, -1 means no limit.
            signature: Whether to show the signature of the Module.
            param_types: Whether to show the types of the parameters.
        Returns:
            A string containing the tabular representation.
        """
        self = to.copy(self)

        if not isinstance(self, tp.Callable):
            raise TypeError(
                "`inputs` can only be specified if the module is a callable."
            )

        with _CallContext(call_info={}):
            assert _CALL_CONTEXT.call_info is not None
            # call using self to preserve references
            def eval_call(args, kwargs):
                initialize = not self.initialized
                with to.make_mutable(self), _ModuleContext(
                    key=utils.Key(42), initializing=initialize
                ):
                    return self(*args, **kwargs)

            jax.eval_shape(eval_call, args, kwargs)

            call_info = _CALL_CONTEXT.call_info

        with to.add_field_info():
            flat: tp.List[to.FieldInfo]
            flat, _ = jax.tree_flatten(self)
            tree_part_types: tp.Tuple[tp.Type[types.TreePart], ...] = tuple(
                {
                    field_info.kind
                    for field_info in flat
                    if utils._generic_issubclass(field_info.kind, types.TreePart)
                }
            )

        path = ()
        rows = list(
            utils._get_tabulate_rows(
                path,
                self,
                summary_depth,
                tree_part_types,
                show_signatures,
                show_param_types,
            )
        )

        modules = [row[0] for row in rows]
        rows = [row[1:] for row in rows]

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

                inputs_repr = utils._format_param_tree(simplified_inputs)
                outputs_repr = utils._format_param_tree(outputs)
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
                utils._format_obj_size(self.filter(kind), add_padding=True)
                for kind in tree_part_types
            ]
        )
        utils._add_padding(rows)

        table = Table(
            show_header=True,
            show_lines=True,
            show_footer=True,
            # box=rich.box.HORIZONTALS,
        )

        table.add_column("path")
        table.add_column("module")
        table.add_column("params")
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
        table.caption = "\nTotal Parameters: " + utils._format_obj_size(
            self, add_padding=False
        )

        return utils._get_rich_repr(table)

    def as_logs(self) -> tp.Dict[str, jnp.ndarray]:
        """
        Returns a flat dictionary with all the non-`Nothing` leafs of the module.
        You can use it to easy gather and report intermediate values published by the module.

        Example:

        ```python
        losses_logs = module.filter(tx.LossLog).as_logs()
        metric_logs = module.filter(tx.MetricLog).as_logs()
        logs = {**losses_logs, **metric_logs}
        ```

        Returns:
            A flat dictionary with all the non-`Nothing` leafs of the module.
        """
        names: tp.Set[str] = set()

        def _get_name(field_info: to.FieldInfo) -> str:
            return (
                field_info.value.name
                if isinstance(field_info.value, types.Named)
                else field_info.name
                if field_info.name is not None
                else field_info.module.__class__.__name__
                if field_info.module is not None
                else "unknown"
            )

        with to.add_field_info():
            fields_info: tp.List[to.FieldInfo] = jax.tree_flatten(
                self,
                is_leaf=lambda x: isinstance(x, types.Named)
                and not isinstance(x.value, to.Nothing),
            )[0]

        # pretend Named values are leaves
        for i, x in enumerate(fields_info):
            if isinstance(x, types.Named):
                field_info = x.value
                field_info.value = types.Named(x.name, field_info.value)
                fields_info[i] = field_info

        logs = {
            _get_name(field_info): field_info.value.value
            if isinstance(field_info.value, types.Named)
            else field_info.value
            for field_info in fields_info
        }
        logs = {utils._unique_name(names, name): value for name, value in logs.items()}

        return logs


# -----------------------------------------------------------------------
# API
# -----------------------------------------------------------------------


def compact_module(f) -> type:
    """
    A decorator that enable the definition of functional Modules
    """
    name = utils._get_name(f)

    @functools.wraps(f)
    @to.compact
    def __call__(self, *args, **kwargs):
        return f(*args, **kwargs)

    module_class = type(
        name,
        (Module,),
        dict(
            __call__=__call__,
        ),
    )

    return module_class


def preserve_state(
    transformation: C, *transformation_args, **transformation_kwargs
) -> C:
    """
    Takes in a function transformation such as `jit` or `vmap` and the function `f`
    to be transformed and returns a the transformed function with the expected behaviour
    but that additionally preserves the state of the first argument of `f`.

    For example, within a `Module`  if you try to `vmap` over a method with stateful operations like this:

    ```python
    @jax.vmap
    def __call__(self, x):
        self.n += 1
        return 2.0 * x
    ```

    It will not work since the composed function `vmap(__call__)` is a pure function so any change
    to `self` will not be reflected outside.


    To solve you can wrap `vmap` using `preserve_state` like this:

    ```python
    @preserve_state(jax.vmap)
    def __call__(self, x):
        self.n += 1
        return 2.0 * x
    ```

    This will guarantee that the state of `self` is propagated to the outside.

    Arguments:
        transformation: The transformation to be applied to the function `f`.
        f: The function to be transformed.
        *args: Additional arguments to be passed to the transformation.
        **kwargs: Additional keyword arguments to be passed to the transformation.

    Returns:
        The transformed function.
    """

    @functools.wraps(transformation)
    def new_transformation(f):
        f_original = f

        f = _return_first(f)
        f = _update_first(
            transformation(f, *transformation_args, **transformation_kwargs)
        )

        @functools.wraps(f_original)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper

    return new_transformation


def _return_first(f):
    def wrapper(self, *args, **kwargs):
        y = f(self, *args, **kwargs)
        return self, y

    return wrapper


def _update_first(f):
    def wrapper(self, *args, **kwargs):
        module, y = f(self, *args, **kwargs)

        self.__dict__.update(module.__dict__)

        return y

    return wrapper


def next_key() -> jnp.ndarray:
    """
    Returns the next key.

    Returns:
        The next key.
    """
    key: jnp.ndarray

    if _MODULE_CONTEXT.key is None:
        raise RuntimeError(
            "RNG key not set, you are either calling an uninitialized Module outside `.init` / `.apply`,"
            "or forgot to call the `rng_key` context manager."
        )

    key, _MODULE_CONTEXT.key = utils.iter_split(_MODULE_CONTEXT.key)

    return key


@contextmanager
def rng_key(key: types.KeyLike):
    key = utils.Key(key)

    with _MODULE_CONTEXT.update(key=key):
        yield
