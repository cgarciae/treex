import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import linear as flax_module

from treex import types
from treex.module import Module
from treex.nn.linear import Linear


class MLP(Module):
    """A Multi-Layer Perceptron (MLP) that applies a sequence of linear layers
    with a given activation (relu by default), the last layer is linear.
    """

    # pytree
    layers: tp.List[Linear]

    # props
    features: tp.Sequence[int]
    module: flax_module.Dense

    def __init__(
        self,
        features: tp.Sequence[int],
        activation: tp.Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
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
            features: a sequence of L+1 integers, where L is the number of layers,
                the first integer is the number of input features and all subsequent
                integers are the number of output features of the respective layer.
            activation: the activation function to use.
            use_bias: whether to add a bias to the output (default: True).
            dtype: the dtype of the computation (default: float32).
            precision: numerical precision of the computation see `jax.lax.Precision`
                for details.
            kernel_init: initializer function for the weight matrix.
            bias_init: initializer function for the bias.
        """

        if len(features) < 2:
            raise ValueError("features must have at least 2 elements")

        self.features = features
        self.activation = activation
        self.layers = [
            Linear(
                features_in=features_in,
                features_out=features_out,
                use_bias=use_bias,
                dtype=dtype,
                precision=precision,
                kernel_init=kernel_init,
                bias_init=bias_init,
            )
            for features_in, features_out in zip(features[:-1], features[1:])
        ]

    def __call__(self, x: np.ndarray) -> jnp.ndarray:
        """
        Applies the MLP to the input.

        Arguments:
            x: input array.

        Returns:
            The output of the MLP.
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)
