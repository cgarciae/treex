import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import stochastic as flax_module

from treex import types
from treex.module import Module
from treex.nn.flax_module import FlaxModule
from treex.rnq_seq import RngSeq


class Dropout(FlaxModule):
    """Create a dropout layer.

    `Dropout` is implemented as a wrapper over `flax.linen.Dropout`, its constructor
    arguments accept almost the same arguments including any Flax artifacts such as initializers.
    Main differences:

    * `deterministic` is not a constructor argument, but remains a `__call__` argument.
    * `self.training` state is used to indicate how Dropout should behave, interally
    `deterministic = not self.training` is used unless `deterministic` is explicitly
    passed via `__call__`.
    * Dropout maintains an `rng: Rng` state which is used to generate random masks unless `rng` is passed
    via `__call__`.
    """

    def __init__(
        self,
        rate: float,
        broadcast_dims: tp.Sequence[int] = (),
    ):
        """
        Create a dropout layer.

        Arguments:
            rate: the dropout probability.  (_not_ the keep rate!)
            broadcast_dims: dimensions that will share the same dropout mask
        """
        if len(broadcast_dims) > 0:
            ndims = max(broadcast_dims) + 1
        else:
            ndims = 1

        shape = list(range(1, ndims + 1))

        super().__init__(
            module=flax_module.Dropout(
                rate=rate,
                broadcast_dims=broadcast_dims,
                deterministic=None,
            ),
            sample_inputs=types.Inputs(
                jnp.ones(shape=shape, dtype=jnp.float32),
                False,
                jax.random.PRNGKey(0),
            ),
            mutable=[],
            rngs=[],
        )

    def __call__(
        self, x: np.ndarray, deterministic: tp.Optional[bool] = None, rng=None
    ) -> jnp.ndarray:
        """Applies a random dropout mask to the input.

        Arguments:
            x: the inputs that should be randomly masked.
            deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
                masked, whereas if true, no mask is applied and the inputs are returned
                as is.
            rng: an optional `jax.random.PRNGKey`. By default `self.rng` will
                be used.

        Returns:
            The masked inputs reweighted to preserve mean.
        """
        training = not deterministic if deterministic is not None else self.training
        deterministic = not training

        if rng is None:
            rng = self.rng_seq.next()

        return super().__call__(x, deterministic=deterministic, rng=rng)
