import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from treex import types
from treex.module import Module


class RngSeq(Module):
    """RNGSeq is simple module that can produce a sequence of PRNGKeys.

    Example:
    ```python
    class Dropout(Module):
        rng: RNGSeq()

        def __init__(self, rate: float):
            self.rng = RNGSeq()
            ...

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            key = self.rng.next()
            mask = jax.random.bernoulli(key, 1.0 - self.rate)
            ...
    ```
    """

    key: types.Rng[types.Initializer, jnp.ndarray]

    def __init__(self, key: tp.Optional[tp.Union[jnp.ndarray, int]] = None):
        """
        Arguments:
            key: An optional PRNGKey to initialize the RNGSeq with.
        """
        super().__init__()
        self.key = (
            jax.random.PRNGKey(key)
            if isinstance(key, int)
            else key
            if isinstance(key, (jnp.ndarray, np.ndarray))
            else types.Initializer(lambda key: key)
        )

    def next(self) -> jnp.ndarray:
        """
        Return a new PRNGKey and updates the internal rng state.

        Returns:
            A PRNGKey.
        """
        assert isinstance(self.key, jnp.ndarray)
        key, self.key = jax.random.split(self.key)
        return key
