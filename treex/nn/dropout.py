import typing as tp
from flax.linen import stochastic as flax_module
import jax
import jax.numpy as jnp
import numpy as np

from treex.module import Module
from treex import types


class Dropout(Module):
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

    rng: types.Rng

    def __init__(
        self,
        rate: float,
        broadcast_dims: tp.Iterable[int] = (),
    ):
        """
        Create a dropout layer.

        Arguments:
            rate: the dropout probability.  (_not_ the keep rate!)
            broadcast_dims: dimensions that will share the same dropout mask
        """
        self.module = flax_module.Dropout(
            rate=rate,
            broadcast_dims=broadcast_dims,
            deterministic=None,
        )
        self.rng = types.Initializer(lambda key: key)

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
        variables = dict()

        training = not deterministic if deterministic is not None else self.training
        deterministic = not training

        if rng is None:
            assert isinstance(self.rng, jnp.ndarray)
            rng, self.rng = jax.random.split(self.rng)

        # call apply
        output = self.module.apply(
            variables,
            x,
            deterministic=deterministic,
            rng=rng,
        )

        return tp.cast(jnp.ndarray, output)
