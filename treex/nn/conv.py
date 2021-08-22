import typing as tp
from flax.linen import linear as flax_module
import jax
import jax.numpy as jnp
import numpy as np

from treex.module import Module
from treex import types


class Conv(Module):
    """Convolution Module wrapping lax.conv_general_dilated.

    `Conv` is implemented as a wrapper over `flax.linen.Conv`, its constructor
    arguments accept almost the same arguments including any Flax artifacts such as initializers.
    Main differences:

    * receives `features_in` as a first argument since shapes must be statically known.
    * `features` argument is renamed to `features_out`.
    """

    # pytree
    params: tp.Optional[tp.Mapping[str, types.Parameter]]

    # props
    features_in: int
    module: flax_module.Conv

    def __init__(
        self,
        features_in: int,
        features_out: int,
        kernel_size: tp.Union[int, tp.Iterable[int]],
        strides: tp.Optional[tp.Iterable[int]] = None,
        padding: tp.Union[str, tp.Iterable[tp.Tuple[int, int]]] = "SAME",
        input_dilation: tp.Optional[tp.Iterable[int]] = None,
        kernel_dilation: tp.Optional[tp.Iterable[int]] = None,
        feature_group_count: int = 1,
        use_bias: bool = True,
        dtype: flax_module.Dtype = jnp.float32,
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
            features_out: number of convolution filters.
            kernel_size: shape of the convolutional kernel. For 1D convolution,
                the kernel size can be passed as an integer. For all other cases, it must
                be a sequence of integers.
            strides: a sequence of `n` integers, representing the inter-window
                strides.
            padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
                of `n` `(low, high)` integer pairs that give the padding to apply before
                and after each spatial dimension.
            input_dilation: `None`, or a sequence of `n` integers, giving the
                dilation factor to apply in each spatial dimension of `inputs`.
                Convolution with input dilation `d` is equivalent to transposed
                convolution with stride `d`.
            kernel_dilation: `None`, or a sequence of `n` integers, giving the
                dilation factor to apply in each spatial dimension of the convolution
                kernel. Convolution with kernel dilation is also known as 'atrous
                convolution'.
            feature_group_count: integer, default 1. If specified divides the input
                features into groups.
            use_bias: whether to add a bias to the output (default: True).
            dtype: the dtype of the computation (default: float32).
            precision: numerical precision of the computation see `jax.lax.Precision`
                for details.
            kernel_init: initializer for the convolutional kernel.
            bias_init: initializer for the bias.
        """
        self.features_in = features_in
        self.module = flax_module.Conv(
            features=features_out,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            feature_group_count=feature_group_count,
            use_bias=use_bias,
            dtype=dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
        self.params = None

    def module_init(self, key: jnp.ndarray):
        if isinstance(self.module.kernel_size, int):
            ndim = 1
            mindim = self.module.kernel_size
        else:
            ndim = len(list(self.module.kernel_size))
            mindim = min(self.module.kernel_size)

        mindim *= 2

        shape = list(range(mindim, mindim + ndim + 1))
        shape[-1] = self.features_in

        x = jax.random.uniform(key, shape=shape)

        variables = self.module.init(key, x)

        # Extract collections
        self.params = variables["params"]

    def __call__(self, x: np.ndarray) -> jnp.ndarray:
        """Applies a convolution to the inputs.

        Arguments:
            x: input data with dimensions (batch, spatial_dims..., features).

        Returns:
            The convolved data.
        """
        assert self.params is not None, "Module not initialized"

        variables = dict(params=self.params)
        output = self.module.apply(variables, x)
        return tp.cast(jnp.ndarray, output)
