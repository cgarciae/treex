import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from treex import types, utils
from treex.module import Module


class KeySeq:
    """KeySeq is simple module that can produce a sequence of PRNGKeys.

    Example:
    ```python
    class Dropout(Module):
        rng: KeySeq()

        def __init__(self, rate: float):
            self.next_key = KeySeq()
            ...

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            key = self.next_key()
            mask = jax.random.bernoulli(key, 1.0 - self.rate)
            ...
    ```
    """

    key: jnp.ndarray

    def __init__(
        self,
        key: tp.Union[jnp.ndarray, int],
    ):
        """
        Arguments:
            key: An optional PRNGKey to initialize the KeySeq with.
        """

        self.key = utils.Key(key)

    def next(self) -> jnp.ndarray:
        """
        Return a new PRNGKey and updates the internal rng state.

        Returns:
            A PRNGKey.
        """

        key, self.key = utils.iter_split(self.key)

        return key

    __next__ = next
