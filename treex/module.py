import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

from treex import types
from treex.tree_object import TreeObject, module_map

A = tp.TypeVar("A")
B = tp.TypeVar("B")
M = tp.TypeVar("M", bound="Module")


class Module(TreeObject):
    _training: bool
    _initialized: bool

    def __init__(self) -> None:
        self._training = True
        self._initialized = False
        super().__init__()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def training(self) -> bool:
        return self._training

    def init(self: M, key: tp.Union[int, jnp.ndarray], inplace: bool = False) -> M:
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

        def next_key() -> jnp.ndarray:
            nonlocal key
            assert isinstance(key, jnp.ndarray)
            next_key, key = jax.random.split(key)
            return next_key

        def call_module_init(module: TreeObject) -> TreeObject:
            if isinstance(module, Module) and not module._initialized:
                module.module_init(next_key())
                module._initialized = True

            return module

        module = jax.tree_map(
            lambda initializer: (
                initializer(next_key())
                if isinstance(initializer, types.Initializer)
                else initializer
            ),
            self,
        )
        if inplace:
            # here we update initialized fields by the above tree_map
            self.update(module, inplace=True)
            # now call module_init inplace
            return module_map(call_module_init, self, inplace=True)
        else:
            return module_map(call_module_init, module, inplace=False)

    def train(self: M, mode: bool = True, inplace: bool = False) -> M:
        """
        Creates a new module with the same structure, but with `TreeObject.training` set to the given value.

        Arguments:
            mode: The new training mode.
        Returns:
            The new module in with the training mode is set to the given value.
        """

        def set_training(module: TreeObject) -> TreeObject:
            if isinstance(module, Module):
                module._training = mode

            return module

        return module_map(set_training, self, inplace=inplace)

    def eval(self: M, inplace: bool = False) -> M:
        """
        Creates a new module with the training mode set to False, equivalent to calling `train(False)`.

        Returns:
            The new module with the training mode set to False.
        """
        return self.train(False, inplace=inplace)
