import dataclasses
import enum
import functools
import inspect
import io
import threading
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import treeo as to
from rich.table import Table
from rich.text import Text

from treex import types, utils

A = tp.TypeVar("A")
B = tp.TypeVar("B")
Filter = tp.Union[
    tp.Type[tp.Type[tp.Any]],
    tp.Callable[[to.FieldInfo], bool],
]
A = tp.TypeVar("A")
B = tp.TypeVar("B")
M = tp.TypeVar("M", bound="Module")


@dataclass
class _Context(threading.local):
    call_info: tp.Optional[tp.Dict["Module", tp.Tuple[types.Inputs, tp.Any]]] = None

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
# Module
# -----------------------------------------


class ModuleMeta(to.TreeMeta):
    def __call__(cls, *args, **kwargs) -> "Module":
        obj = tp.cast(Module, super().__call__(*args, **kwargs))

        if isinstance(obj, tp.Callable):
            new_call = _get_call(cls.__call__)
            cls.__call__ = functools.wraps(cls.__call__)(new_call)

        return obj


# override __call__
def _get_call(orig_call):
    def _meta_call(self: Module, *args, **kwargs):
        outputs = orig_call(self, *args, **kwargs)

        if _CONTEXT.call_info is not None:
            inputs = types.Inputs(*args, **kwargs)
            _CONTEXT.call_info[self] = (inputs, outputs)

        return outputs

    return _meta_call


class Module(to.Tree, metaclass=ModuleMeta):
    # use to.field to copy class vars to instance
    _training: bool = to.static(True)
    _initialized: bool = to.static(False)
    _frozen: bool = to.static(False)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def training(self) -> bool:
        return self._training

    @property
    def frozen(self) -> bool:
        return self._frozen

    def init(self: M, key: tp.Union[int, jnp.ndarray], inplace: bool = False) -> M:
        """
        Creates a new module with the same structure, but with its fields initialized given a seed `key`. The following
        procedure is used:

        1. The input `key` is split and iteratively updated before passing a derived value to any
            process that requires initialization.
        2. `Initializer`s are called and applied to the module first.
        3. `Module.module_init` methods are called last.

        Arguments:
            key: The seed to use for initialization.
        Returns:
            The new module with the fields initialized.
        """
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)

        def next_key() -> jnp.ndarray:
            nonlocal key
            assert isinstance(key, jnp.ndarray)
            next_key, key = jax.random.split(key)
            return next_key

        module: M = jax.tree_map(
            lambda initializer: (
                initializer(next_key())
                if isinstance(initializer, types.Initializer)
                else initializer
            ),
            self,
            is_leaf=lambda x: isinstance(x, types.Initializer),
        )

        def call_module_init(module: Module):
            if isinstance(module, Module) and not module._initialized:
                module.module_init(next_key())
                module._initialized = True

        if inplace:
            # here we update initialized fields by the above tree_map
            module = self.update(module, inplace=True)

        return to.apply(call_module_init, module, inplace=inplace)

    def train(self: M, mode: bool = True, inplace: bool = False) -> M:
        """
        Creates a new module with the same structure, but with `Module.training` set to the given value.

        Arguments:
            mode: The new training mode.
            inplace: Whether to update the module inplace.
        Returns:
            The new module in with the training mode is set to the given value,
            if `inplace` is `True` then `self` is returned.
        """

        def set_training(module: Module):
            if isinstance(module, Module):
                module._training = mode

        return to.apply(set_training, self, inplace=inplace)

    def eval(self: M, inplace: bool = False) -> M:
        """
        Creates a new module with the training mode set to False, equivalent to calling `train(False)`.

        Returns:
            The new module with the training mode set to False.
        """
        return self.train(False, inplace=inplace)

    def freeze(self: M, mode: bool = True, inplace: bool = False) -> M:
        """
        Creates a new module with the same structure, but with `Module.frozen` set
        to the given value.

        Arguments:
            mode: The new `frozen` mode.
            inplace: Whether to update the module inplace.
        Returns:
            The new module in with the `frozen` mode is set to the given value,
            if `inplace` is `True` then `self` is returned.
        """

        def set_frozen(module: Module):
            if isinstance(module, Module):
                module._frozen = mode

        return to.apply(set_frozen, self, inplace=inplace)

    def unfreeze(self: M, inplace: bool = False) -> M:
        """
        Creates a new module with `.frozen` set to False, equivalent to calling `freeze(False)`.

        Arguments:
            inplace: Whether to update the module inplace.
        Returns:
            The new module with `.frozen` set to False, if `inplace` is `True` then `self` is returned.
        """
        return self.freeze(False, inplace=inplace)

    def module_init(self, key: jnp.ndarray) -> None:
        pass

    def map(self: M, f: tp.Callable, *filters: Filter, inplace: bool = False) -> M:
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

    def filter(self: M, *filters: Filter) -> M:
        """
        Creates a new module with the same structure, but sets to `Nothing` leaves that
        do not match **all** of the given filters. If a type `t` is given, the filter

        ```python
        _filter(x: Type[types.TreePart]) = issubclass(x, t)
        ```

        will be created for that type.

        Example:
        ```python
        class MyModule(tx.Module):
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

    def update(self: M, other: M, *rest: M, inplace: bool = False) -> M:
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
            signature: Whether to show the signature of the Module.
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
                    if utils._generic_issubclass(field_info.kind, types.TreePart)
                }
            )

        path = ()
        rows = list(
            utils._get_tabulate_rows(
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
        table.caption = "\nTotal Parameters: " + utils._format_obj_size(
            self, add_padding=False
        )

        return utils._get_rich_repr(table)

    def __repr__(self) -> str:
        rep = utils._get_repr(self, level=0, array_type=None, inline=False)
        return utils._get_rich_repr(Text.from_markup(rep))

    # --------------------------------------
    # filter shortcuts
    # --------------------------------------

    def parameters(self: M, *filters: Filter) -> M:
        """
        Returns a copy of the Module with only tx.Parameter TreeParts, alias for `filter(tx.Parameter)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.Parameter, *filters)

    def batch_stats(self: M, *filters: Filter) -> M:
        """
        Returns a copy of the Module with only tx.BatchStat TreeParts, alias for `filter(tx.BatchStat)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.BatchStat, *filters)

    def rngs(self: M, *filters: Filter) -> M:
        """
        Returns a copy of the Module with only tx.Rng TreeParts, alias for `filter(tx.Rng)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.Rng, *filters)

    def model_states(self: M, *filters: Filter) -> M:
        """
        Returns a copy of the Module with only tx.ModelState TreeParts, alias for `filter(tx.ModelState)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.ModelState, *filters)

    def states(self: M, *filters: Filter) -> M:
        """
        Returns a copy of the Module with only tx.State TreeParts, alias for `filter(tx.State)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.State, *filters)

    def metrics(self: M, *filters: Filter) -> M:
        """
        Returns a copy of the Module with only tx.Metric TreeParts, alias for `filter(tx.Metric)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.Metric, *filters)

    def losses(self: M, *filters: Filter) -> M:
        """
        Returns a copy of the Module with only tx.Loss TreeParts, alias for `filter(tx.Loss)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.Loss, *filters)

    def logs(self: M, *filters: Filter) -> M:
        """
        Returns a copy of the Module with only tx.Log TreeParts, alias for `filter(tx.Log)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.filter(types.Log, *filters)


# --------------------------------------------------
# utils
# --------------------------------------------------
