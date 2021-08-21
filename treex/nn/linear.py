import typing as tp
from flax.linen import linear as flax_module
import jax
import jax.numpy as jnp
import numpy as np

from treex.module import Module
from treex import types


class Linear(Module):
    """A linear transformation applied over the last dimension of the input.

    `Linear` is implemented as a wrapper over `flax.linen.Dense`, its constructor
    arguments accept the same flax artifacts.
    """

    params: tp.Dict[str, types.Parameter]

    def __init__(
        self,
        features_in: int,
        features_out: int,
        use_bias: bool = True,
        dtype: tp.Any = jnp.float32,
        precision: tp.Any = None,
        kernel_init: tp.Callable[
            [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype],
            flax_module.Array,
        ] = flax_module.default_kernel_init,
        bias_init: tp.Callable[
            [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype],
            flax_module.Array,
        ] = flax_module.zeros,
    ):
        """
        Arguments:
            features_in: the number of input features.
            features_out: the number of output features.
            use_bias: whether to add a bias to the output (default: True).
            dtype: the dtype of the computation (default: float32).
            precision: numerical precision of the computation see `jax.lax.Precision`
                for details.
            kernel_init: initializer function for the weight matrix.
            bias_init: initializer function for the bias.
        """
        self.module = flax_module.Dense(
            features=features_out,
            use_bias=use_bias,
            dtype=dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
        self.params = types.Initializer(lambda key: self._flax_init(key, features_in))

    def post_init(self):
        assert isinstance(self.params, tp.Mapping)

        # variables was temporarily stored in params during init
        variables = self.params

        # Extract collections
        self.params = variables["params"]

    def __call__(self, x: np.ndarray) -> jnp.ndarray:
        variables = dict(params=self.params)
        output = self.module.apply(variables, x)
        return tp.cast(jnp.ndarray, output)

    def _flax_init(
        self,
        key: jnp.ndarray,
        features_in,
    ):
        batch_size = 10  # random
        x = jax.random.uniform(key, (batch_size, features_in))

        variables = self.module.init(key, x)

        return variables
