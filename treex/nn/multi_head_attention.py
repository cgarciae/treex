import typing as tp

import einops
import flax.linen.attention as flax_module
import jax
import jax.numpy as jnp
import treeo as to

from treex import types
from treex.module import Module, preserve_state
from treex.nn import dropout
from treex.nn.dropout import Dropout
from treex.nn.linear import Linear

Shape = tp.Sequence[int]

FlaxInitializer = tp.Callable[
    [flax_module.PRNGKey, Shape, flax_module.Dtype],
    flax_module.Array,
]


class Attention(Module):

    query_kernel: tp.Union[jnp.ndarray, types.Initializer] = types.Parameter.node()
    key_kernel: tp.Union[jnp.ndarray, types.Initializer] = types.Parameter.node()
    value_kernel: tp.Union[jnp.ndarray, types.Initializer] = types.Parameter.node()
    projection_kernel: tp.Union[jnp.ndarray, types.Initializer] = types.Parameter.node()
    projection_bias: tp.Optional[
        tp.Union[jnp.ndarray, types.Initializer]
    ] = types.Parameter.node()

    def __init__(
        self,
        num_heads: int,
        *,
        head_size: tp.Optional[int] = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        kernel_initializer: FlaxInitializer = flax_module.default_kernel_init,
        bias_initializer: FlaxInitializer = flax_module.zeros,
        axis_name: str = "head",
    ):

        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_rate = dropout
        self.use_projection_bias = use_projection_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.axis_name = axis_name

    @staticmethod
    @to.compact
    def __call__(
        module,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: tp.Optional[jnp.ndarray],
    ) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:

        head_size = (
            module.head_size if module.head_size is not None else query.shape[-1]
        )

        # Linear transformations
        query = Linear(
            head_size,
            kernel_init=module.kernel_initializer,
            bias_init=module.bias_initializer,
            use_bias=module.use_projection_bias,
            axis_name=module.axis_name,
        )(query)
        key = Linear(
            head_size,
            kernel_init=module.kernel_initializer,
            bias_init=module.bias_initializer,
            use_bias=module.use_projection_bias,
            axis_name=module.axis_name,
        )(key)
        value = Linear(
            head_size,
            kernel_init=module.kernel_initializer,
            bias_init=module.bias_initializer,
            use_bias=module.use_projection_bias,
            axis_name=module.axis_name,
        )(value)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        query /= jnp.sqrt(query.shape[-1])

        # Calculate dot product attention
        logits = jnp.einsum("...ND, ...MD -> ...NM", query, key)

        # apply mask
        if mask is not None:
            logits += -10e9 * (1.0 - mask)

        attention_matrix = jax.nn.softmax(logits)

        attention_dropout = Dropout(
            rate=module.dropout_rate,
            axis_name=module.axis_name,
        )(attention_matrix)

        # attention * value
        output = jnp.einsum("...NM, ...MD -> ...ND", attention_dropout, value)

        return output, attention_matrix


class MultiHeadAttention(Module):
    def __init__(
        self,
        num_heads: int,
        *,
        head_size: tp.Optional[int] = None,
        output_size: tp.Optional[int] = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        kernel_initializer: FlaxInitializer = flax_module.default_kernel_init,
        bias_initializer: FlaxInitializer = flax_module.zeros,
        axis_name: str = "head",
    ):

        self.num_heads = num_heads
        self.head_size = head_size
        self.output_size = output_size
        self.dropout_rate = dropout
        self.use_projection_bias = use_projection_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.axis_name = axis_name

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

    @to.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: tp.Optional[jnp.ndarray] = None,
        value: tp.Optional[jnp.ndarray] = None,
        *,
        mask: tp.Optional[jnp.ndarray] = None,
        return_attn_coef: bool = False,
    ) -> tp.Union[jnp.ndarray, tp.Tuple[jnp.ndarray, jnp.ndarray]]:

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

        attention = Attention(
            num_heads=self.num_heads,
            head_size=self.head_size,
            dropout=self.dropout_rate,
            use_projection_bias=self.use_projection_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            axis_name=self.axis_name,
        )

        multi_head_attention = preserve_state(
            jax.vmap,
            in_axes=(0, 0, 0, 0, mask_axis),
            out_axes=(-1, 0),
            axis_name=self.axis_name,
        )(attention)

        multihead_output, multihead_attention_matrix = multi_head_attention(
            attention,
            query,
            key,
            value,
            mask,
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
