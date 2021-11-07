import typing as tp

import einops
import flax.linen.attention as flax_module
import jax
import jax.numpy as jnp
import treeo as to

from treex import types
from treex.module import Module
from treex.nn import dropout
from treex.nn.dropout import Dropout

Shape = tp.Sequence[int]

FlaxInitializer = tp.Callable[
    [flax_module.PRNGKey, Shape, flax_module.Dtype],
    flax_module.Array,
]


class MultiHeadParams(tp.NamedTuple):
    query_kernel: tp.Union[jnp.ndarray, types.Initializer]
    key_kernel: tp.Union[jnp.ndarray, types.Initializer]
    value_kernel: tp.Union[jnp.ndarray, types.Initializer]


class MultiHeadAttention(Module):
    r"""
    MultiHead Attention layer.
    Defines the MultiHead Attention operation as described in
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762) which takes
    in the tensors `query`, `key`, and `value`, and returns the dot-product attention
    between them:

    ```python
    mha = MultiHeadAttention(head_size=128, num_heads=12)
    query = tf.random.uniform((32, 20, 200)) # (batch_size, query_elements, query_depth)
    key = tf.random.uniform((32, 15, 300)) # (batch_size, key_elements, key_depth)
    value = tf.random.uniform((32, 15, 400)) # (batch_size, key_elements, value_depth)
    attention = mha([query, key, value]) # (batch_size, query_elements, value_depth)
    ```

    If `value` is not given then internally `value = key` will be used:

    ```python
    mha = MultiHeadAttention(head_size=128, num_heads=12)
    query = tf.random.uniform((32, 20, 200)) # (batch_size, query_elements, query_depth)
    key = tf.random.uniform((32, 15, 300)) # (batch_size, key_elements, key_depth)
    attention = mha([query, key]) # (batch_size, query_elements, key_depth)
    ```

    Arguments:
        input_size: The size of the input tensors.
        head_size: int, dimensionality of the `query`, `key` and `value` tensors
        after the linear transformation.
        num_heads: int, number of attention heads.
        output_size: int, dimensionality of the output space, if `None` then the
        input dimension of
        `value` or `key` will be used, default `None`.
        dropout: float, `rate` parameter for the dropout layer that is
        applied to attention after softmax,
        default `0`.
        use_projection_bias: bool, whether to use a bias term after the linear
        output projection.
        an additional output argument.
        kernel_initializer: initializer, initializer for the kernel weights.
        kernel_regularizer: regularizer, regularizer for the kernel weights.
        kernel_constraint: constraint, constraint for the kernel weights.
        bias_initializer: initializer, initializer for the bias weights.
        bias_regularizer: regularizer, regularizer for the bias weights.
        bias_constraint: constraint, constraint for the bias weights.
    """

    query_kernel: tp.Union[jnp.ndarray, types.Initializer] = types.Parameter.node()
    key_kernel: tp.Union[jnp.ndarray, types.Initializer] = types.Parameter.node()
    value_kernel: tp.Union[jnp.ndarray, types.Initializer] = types.Parameter.node()
    projection_kernel: tp.Union[jnp.ndarray, types.Initializer] = types.Parameter.node()
    projection_bias: tp.Optional[
        tp.Union[jnp.ndarray, types.Initializer]
    ] = types.Parameter.node()

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        *,
        head_size: tp.Optional[int] = None,
        output_size: tp.Optional[int] = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        kernel_initializer: FlaxInitializer = flax_module.default_kernel_init,
        bias_initializer: FlaxInitializer = flax_module.zeros,
        # kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
        # kernel_regularizer: typing.Union[str, typing.Callable] = None,
        # kernel_constraint: typing.Union[str, typing.Callable] = None,
        # bias_regularizer: typing.Union[str, typing.Callable] = None,
        # bias_constraint: typing.Union[str, typing.Callable] = None,
    ):
        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.input_size = input_size
        self.num_heads = num_heads
        self.head_size = head_size if head_size is not None else input_size
        self.output_size = output_size if output_size is not None else input_size
        self.use_projection_bias = use_projection_bias
        self.droput_rate = dropout

        # submodules
        self.dropout = Dropout(self.droput_rate)

        # nodes
        self.query_kernel = types.Initializer(
            lambda key: kernel_initializer(
                key, [self.num_heads, self.input_size, self.head_size], jnp.float32
            )
        )
        self.key_kernel = types.Initializer(
            lambda key: kernel_initializer(
                key, [self.num_heads, self.input_size, self.head_size], jnp.float32
            )
        )
        self.value_kernel = types.Initializer(
            lambda key: kernel_initializer(
                key, [self.num_heads, self.input_size, self.head_size], jnp.float32
            )
        )
        self.projection_kernel = types.Initializer(
            lambda key: kernel_initializer(
                key, [self.num_heads * self.head_size, self.output_size], jnp.float32
            )
        )
        self.projection_bias = (
            types.Initializer(
                lambda key: bias_initializer(key, [self.output_size], jnp.float32)
            )
            if self.use_projection_bias
            else None
        )

    @tp.overload
    def __call__(
        self,
        query: jnp.ndarray,
        key: tp.Optional[jnp.ndarray] = None,
        value: tp.Optional[jnp.ndarray] = None,
        *,
        mask: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        ...

    @tp.overload
    def __call__(
        self,
        query: jnp.ndarray,
        key: tp.Optional[jnp.ndarray] = None,
        value: tp.Optional[jnp.ndarray] = None,
        *,
        mask: tp.Optional[jnp.ndarray] = None,
        return_attn_coef: bool = False,
    ) -> tp.Union[jnp.ndarray, tp.Tuple[jnp.ndarray, jnp.ndarray]]:
        ...

    def __call__(
        self,
        query: jnp.ndarray,
        key: tp.Optional[jnp.ndarray] = None,
        value: tp.Optional[jnp.ndarray] = None,
        *,
        mask: tp.Optional[jnp.ndarray] = None,
        return_attn_coef: bool = False,
    ) -> tp.Union[jnp.ndarray, tp.Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Arguments:
            inputs:  List of `[query, key, value]` where
                * `query`: np.ndarray of shape `(..., query_elements, query_depth)`
                * `key`: `np.ndarray of shape '(..., key_elements, key_depth)`
                * `value`: np.ndarray of shape `(..., key_elements, value_depth)`, optional, if not given `key` will be used.
            mask: a binary np.ndarray of shape `[batch_size?, num_heads?, query_elements, key_elements]`
                which specifies which query elements can attendo to which key elements,
                `1` indicates attention and `0` indicates no attention.
        Output shape:
            * `(..., query_elements, output_size)` if `output_size` is given, else
            * `(..., query_elements, value_depth)` if `value` is given, else
            * `(..., query_elements, key_depth)`
        """

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # D = input features
        # O = output features

        if key is None:
            key = query

        if value is None:
            value = key

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )

        mask_axis = None
        if mask is not None:
            if mask.ndim < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

            if mask.ndim == 3:
                if mask.shape[0] != self.num_heads:
                    raise ValueError(
                        "If mask has 3 dimensions, the first dimension must be equal to the number of heads"
                    )
                mask_axis = 0

            mask = mask.astype(jnp.float32)

        multi_head_attention = jax.vmap(
            self.attention,
            in_axes=(0, None, None, None, mask_axis, None),
            out_axes=(-1, -3),
            axis_name="head",
        )

        multihead_output, multihead_attention_matrix = multi_head_attention(
            MultiHeadParams(
                query_kernel=self.query_kernel,
                key_kernel=self.key_kernel,
                value_kernel=self.value_kernel,
            ),
            query,
            key,
            value,
            mask,
            self.dropout,
        )
        multihead_output = einops.rearrange(multihead_output, "... D H-> ... (D H)")

        output = jnp.dot(multihead_output, self.projection_kernel)

        if self.projection_bias is not None:
            assert isinstance(self.projection_bias, jnp.ndarray)
            output += self.projection_bias

        if return_attn_coef:
            return output, multihead_attention_matrix
        else:
            return output

    @staticmethod
    def attention(
        params: MultiHeadParams,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: tp.Optional[jnp.ndarray],
        dropout: Dropout,
    ) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:

        # Linear transformations
        query = jnp.dot(query, params.query_kernel)
        key = jnp.dot(key, params.key_kernel)
        value = jnp.dot(value, params.value_kernel)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        query /= jnp.sqrt(query.shape[-1])

        # Calculate dot product attention
        logits = jnp.einsum("...ND, ...MD -> ...NM", query, key)

        # apply mask
        if mask is not None:
            logits += -10e9 * (1.0 - mask)

        attention_matrix = jax.nn.softmax(logits)

        # attention dropout
        rng = jax.random.fold_in(
            dropout.next_key(),
            jax.lax.axis_index("head"),
        )
        attention_dropout = dropout(attention_matrix, rng=rng)

        # attention * value
        output = jnp.einsum("...NM, ...MD -> ...ND", attention_dropout, value)

        return output, attention_matrix
