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
    arguments accept almost the same arguments including any Flax artifacts such as initializers.
    Main differences:

    * receives `features_in` as a first argument since shapes must be statically known.
    * `features` argument is renamed to `features_out`.
    """

    # pytree
    params: tp.Optional[tp.Mapping[str, types.Parameter]]

    # props
    features_in: int
    module: flax_module.Dense

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
        self.features_in = features_in
        self.module = flax_module.Dense(
            features=features_out,
            use_bias=use_bias,
            dtype=dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
        self.params = None

    def module_init(self, key: jnp.ndarray):
        batch_size = 10  # random
        x = jax.random.uniform(key, (batch_size, self.features_in))

        variables = self.module.init(key, x)

        # Extract collections
        self.params = variables["params"]

    def __call__(self, x: np.ndarray) -> jnp.ndarray:
        """Applies a linear transformation to the inputs along the last dimension.

        Arguments:
            x: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        assert self.params is not None, "Module not initialized"

        variables = dict(params=self.params)
        output = self.module.apply(variables, x)
        return tp.cast(jnp.ndarray, output)
