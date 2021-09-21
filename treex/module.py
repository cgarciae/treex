import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import treeo as to

from treex import types
from treex.treex import ProtoModule

A = tp.TypeVar("A")
B = tp.TypeVar("B")
M = tp.TypeVar("M", bound="Module")


class Module(ProtoModule):
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
        3. `ProtoModule.module_init` methods are called last.

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

        def call_module_init(module: ProtoModule):
            if isinstance(module, Module) and not module._initialized:
                module.module_init(next_key())
                module._initialized = True

        if inplace:
            # here we update initialized fields by the above tree_map
            module = self.update(module, inplace=True)

        return to.apply(call_module_init, module, inplace=inplace)

    def train(self: M, mode: bool = True, inplace: bool = False) -> M:
        """
        Creates a new module with the same structure, but with `ProtoModule.training` set to the given value.

        Arguments:
            mode: The new training mode.
            inplace: Whether to update the module inplace.
        Returns:
            The new module in with the training mode is set to the given value,
            if `inplace` is `True` then `self` is returned.
        """

        def set_training(module: ProtoModule):
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
        Creates a new module with the same structure, but with `ProtoModule.frozen` set
        to the given value.

        Arguments:
            mode: The new `frozen` mode.
            inplace: Whether to update the module inplace.
        Returns:
            The new module in with the `frozen` mode is set to the given value,
            if `inplace` is `True` then `self` is returned.
        """

        def set_frozen(module: ProtoModule):
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
