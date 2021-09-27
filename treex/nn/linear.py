import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import linear as flax_module

from treex import types
from treex.module import Module


class Linear(Module):
    """A linear transformation applied over the last dimension of the input.

    `Linear` is implemented as a wrapper over `flax.linen.Dense`, its constructor
    arguments accept almost the same arguments including any Flax artifacts such as initializers.
    Main differences:

    * receives `features_in` as a first argument since shapes must be statically known.
    * `features` argument is renamed to `features_out`.
    """

    # pytree
    kernel: tp.Optional[jnp.ndarray] = types.Parameter.node()
    bias: tp.Optional[jnp.ndarray] = types.Parameter.node()

    # static
    features_in: int
    features_out: int
    use_bias: bool
    dtype: tp.Any
    precision: tp.Any
    kernel_init: tp.Callable[
        [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype],
        flax_module.Array,
    ]
    bias_init: tp.Callable[
        [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype],
        flax_module.Array,
    ]

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
        self.features_out = features_out
        self.use_bias = use_bias
        self.dtype = dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init

        self.kernel = None
        self.bias = None

    @property
    def module(self) -> flax_module.Dense:
        return flax_module.Dense(
            features=self.features_out,
            use_bias=self.use_bias,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def rng_init(self, key: jnp.ndarray):
        batch_size = 10  # random
        x = jax.random.uniform(key, (batch_size, self.features_in))

        variables = self.module.init({"params": key}, x)

        # Extract collections
        params = variables["params"].unfreeze()

        self.kernel = params["kernel"]

        if self.use_bias:
            self.bias = params["bias"]

    def __call__(self, x: np.ndarray) -> jnp.ndarray:
        """Applies a linear transformation to the inputs along the last dimension.

        Arguments:
            x: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        assert self.initialized, "Module is not initialized."

        assert self.kernel is not None
        params = {"kernel": self.kernel}

        if self.use_bias:
            assert self.bias is not None
            params["bias"] = self.bias

        output = self.module.apply({"params": params}, x)
        return tp.cast(jnp.ndarray, output)
