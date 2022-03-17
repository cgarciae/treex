import typing as tp

import jax.numpy as jnp
from flax.linen import attention as flax_module

from treex import types
from treex.key_seq import KeySeq
from treex.module import Module, next_key

PRNGKey = flax_module.PRNGKey
Shape = flax_module.Shape
Dtype = flax_module.Dtype
Array = flax_module.Array


class MultiHeadDotProductAttention(Module):
    """Multi-head dot product attention.

    `MultiHeadDotProductAttention` is implemented as a wrapper over `flax.linen.MultiHeadDotProductAttention`,
    its constructor arguments accept almost the same arguments including any Flax artifacts such as initializers.
    """

    # pytree
    query: tp.Dict[str, jnp.ndarray] = types.Parameter().node()
    key: tp.Dict[str, jnp.ndarray] = types.Parameter().node()
    value: tp.Dict[str, jnp.ndarray] = types.Parameter().node()
    out: tp.Dict[str, jnp.ndarray] = types.Parameter().node()

    # props
    num_heads: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    qkv_features: tp.Optional[int] = None
    out_features: tp.Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: tp.Optional[bool] = None
    precision: tp.Any = None
    kernel_init: tp.Callable[
        [PRNGKey, Shape, Dtype], Array
    ] = flax_module.default_kernel_init
    bias_init: tp.Callable[[PRNGKey, Shape, Dtype], Array] = flax_module.zeros
    use_bias: bool = True
    attention_fn: tp.Callable[
        [Array, Array, Array], Array
    ] = flax_module.dot_product_attention
    decode: bool = False

    def __init__(
        self,
        num_heads: int,
        dtype: Dtype = jnp.float32,
        param_dtype: Dtype = jnp.float32,
        qkv_features: tp.Optional[int] = None,
        out_features: tp.Optional[int] = None,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: tp.Optional[bool] = None,
        precision: tp.Any = None,
        kernel_init: tp.Callable[
            [PRNGKey, Shape, Dtype], Array
        ] = flax_module.default_kernel_init,
        bias_init: tp.Callable[[PRNGKey, Shape, Dtype], Array] = flax_module.zeros,
        use_bias: bool = True,
        attention_fn: tp.Callable[
            [Array, Array, Array], Array
        ] = flax_module.dot_product_attention,
        decode: bool = False,
        name: tp.Optional[str] = None,
    ):
        """
        Arguments:
          num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1]) should be divisible by the number of heads.
          dtype: the dtype of the computation (default: float32)
          param_dtype: the dtype passed to parameter initializers (default: float32).
          qkv_features: dimension of the key, query, and value.
          out_features: dimension of the last projection
          broadcast_dropout: bool: use a broadcasted dropout along batch dims.
          dropout_rate: dropout rate
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.
          precision: numerical precision of the computation see `jax.lax.Precision`
            for details.
          kernel_init: initializer for the kernel of the Dense layers.
          bias_init: initializer for the bias of the Dense layers.
          use_bias: bool: whether pointwise QKVO dense transforms use bias.
          attention_fn: dot_product_attention or compatible function. Accepts
            query, key, value, and returns output of shape
            `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
          decode: whether to prepare and use an autoregressive cache.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.qkv_features = qkv_features
        self.out_features = out_features
        self.broadcast_dropout = broadcast_dropout
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.use_bias = use_bias
        self.attention_fn = attention_fn
        self.decode = decode
        self.next_key = KeySeq()

    @property
    def module(self) -> flax_module.MultiHeadDotProductAttention:
        return flax_module.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            qkv_features=self.qkv_features,
            out_features=self.out_features,
            broadcast_dropout=self.broadcast_dropout,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            attention_fn=self.attention_fn,
            decode=self.decode,
        )

    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: tp.Optional[Array] = None,
        deterministic: tp.Optional[bool] = None,
        rng=None,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
            applies dot-product attention and project the results to an output vector.

        Arguments:
          inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
          inputs_kv: key/values of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        if self.initializing():
            rngs = {"params": next_key(), "dropout": next_key()}
            variables = self.module.init(rngs, inputs_q, inputs_kv, mask)

            # Extract collections
            params = variables["params"].unfreeze()

            self.query = params["query"]
            self.key = params["key"]
            self.value = params["value"]
            self.out = params["out"]

        assert self.query is not None
        assert self.key is not None
        assert self.value is not None
        assert self.out is not None

        params = {
            "query": self.query,
            "key": self.query,
            "value": self.value,
            "out": self.out,
        }

        training = (
            not deterministic
            if deterministic is not None
            else self.training and not self.frozen
        )

        if rng is None:
            rng = self.next_key() if training else self.next_key.key

        output = self.module.apply(
            {"params": params},
            inputs_q,
            inputs_kv,
            mask,
            deterministic,
            rngs={"dropout": rng},
        )
        return tp.cast(jnp.ndarray, output)
