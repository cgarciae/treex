import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from treex import types, utils
from treex.module import Module


class KeySeq(Module):
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

    key: tp.Union[types.Initializer, jnp.ndarray] = types.Rng.node()

    def __init__(
        self,
        key: tp.Optional[tp.Union[jnp.ndarray, int]] = None,
        *,
        axis_name: tp.Optional[tp.Any] = None
    ):
        """
        Arguments:
            key: An optional PRNGKey to initialize the KeySeq with.
        """

        self.key = (
            utils.Key(key)
            if isinstance(key, int)
            else key
            if isinstance(key, (jnp.ndarray, np.ndarray))
            else types.Initializer(lambda key: key)
        )
        self.axis_name = axis_name

    def __call__(self, *, axis_name: tp.Optional[tp.Any] = None) -> jnp.ndarray:
        """
        Return a new PRNGKey and updates the internal rng state.

        Returns:
            A PRNGKey.
        """
        key: jnp.ndarray

        assert isinstance(self.key, jnp.ndarray)
        key, self.key = utils.iter_split(self.key)

        if axis_name is None:
            axis_name = self.axis_name

        if axis_name is not None:
            axis_index = jax.lax.axis_index(axis_name)
            key = jax.random.fold_in(key, axis_index)

        return key
