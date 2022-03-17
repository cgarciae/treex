import unittest

import hypothesis as hp
import jax
import numpy as np
from flax import linen
from hypothesis import strategies as st

import treex as tx

BIAS_INITS = (
    tx.initializers.zeros,
    tx.initializers.ones,
    tx.initializers.normal(),
    tx.initializers.uniform(),
)
KERNEL_INITS = BIAS_INITS + (
    tx.initializers.xavier_uniform(),
    tx.initializers.xavier_normal(),
    tx.initializers.lecun_uniform(),
    tx.initializers.lecun_normal(),
    tx.initializers.kaiming_uniform(),
    tx.initializers.kaiming_normal(),
)


class MultiHeadDotProductAttentionTest(unittest.TestCase):
    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        length=st.integers(min_value=1, max_value=32),
        log2_features_in=st.integers(min_value=1, max_value=5),
        log2_num_heads=st.integers(min_value=1, max_value=3),
        log2_qkv_features=st.integers(min_value=1, max_value=5),
        log2_out_features=st.integers(min_value=1, max_value=5),
        broadcast_dropout=st.booleans(),
        dropout_rate=st.floats(min_value=0.0, max_value=1.0),
        deterministic=st.booleans(),
        kernel_init=st.sampled_from(KERNEL_INITS),
        bias_init=st.sampled_from(BIAS_INITS),
        use_bias=st.booleans(),
        decode=st.booleans(),
        training=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=1)
    def test_equivalence(
        self,
        batch_size,
        length,
        log2_features_in,
        log2_num_heads,
        log2_qkv_features,
        log2_out_features,
        broadcast_dropout,
        dropout_rate,
        deterministic,
        kernel_init,
        bias_init,
        use_bias,
        decode,
        training,
    ):

        shape = (batch_size, length, 2 ** log2_features_in)

        inputs_q = np.random.uniform(size=shape)
        inputs_kv = np.random.uniform(size=shape)

        key = tx.Key(42)

        flax_module = linen.MultiHeadDotProductAttention(
            num_heads=2 ** log2_num_heads,
            qkv_features=2 ** log2_qkv_features,
            out_features=2 ** log2_out_features,
            broadcast_dropout=broadcast_dropout,
            dropout_rate=dropout_rate,
            deterministic=deterministic,
            kernel_init=kernel_init,
            bias_init=bias_init,
            use_bias=use_bias,
            decode=False,
        )
        treex_module = tx.MultiHeadDotProductAttention(
            num_heads=2 ** log2_num_heads,
            qkv_features=2 ** log2_qkv_features,
            out_features=2 ** log2_out_features,
            broadcast_dropout=broadcast_dropout,
            dropout_rate=dropout_rate,
            deterministic=deterministic,
            kernel_init=kernel_init,
            bias_init=bias_init,
            use_bias=use_bias,
            decode=False,
        ).train(training)

        flax_key, _ = tx.iter_split(key)  # emulate init split
        variables = flax_module.init(key, inputs_q, inputs_kv)
        treex_module = treex_module.init(key, (inputs_q, inputs_kv))

        assert np.allclose(
            variables["params"]["query"]["kernel"], treex_module.query["kernel"]
        )
        assert np.allclose(
            variables["params"]["key"]["kernel"], treex_module.key["kernel"]
        )
        assert np.allclose(
            variables["params"]["value"]["kernel"], treex_module.value["kernel"]
        )
        assert np.allclose(
            variables["params"]["out"]["kernel"], treex_module.out["kernel"]
        )
        if use_bias:
            assert np.allclose(
                variables["params"]["query"]["bias"], treex_module.query["bias"]
            )
            assert np.allclose(
                variables["params"]["key"]["bias"], treex_module.key["bias"]
            )
            assert np.allclose(
                variables["params"]["value"]["bias"], treex_module.value["bias"]
            )
            assert np.allclose(
                variables["params"]["out"]["bias"], treex_module.out["bias"]
            )

        # split key same way tx.Dropout does internally
        rng, _ = tx.iter_split(flax_key, 2)
        y_flax = flax_module.apply(
            variables, rngs={"dropout": rng}, inputs_q=inputs_q, inputs_kv=inputs_kv
        )

        y_treex = treex_module(inputs_q=inputs_q, inputs_kv=inputs_kv)

        assert np.allclose(y_flax, y_treex)

        assert np.allclose(
            variables["params"]["query"]["kernel"], treex_module.query["kernel"]
        )
        assert np.allclose(
            variables["params"]["key"]["kernel"], treex_module.key["kernel"]
        )
        assert np.allclose(
            variables["params"]["value"]["kernel"], treex_module.value["kernel"]
        )
        assert np.allclose(
            variables["params"]["out"]["kernel"], treex_module.out["kernel"]
        )
        if use_bias:
            assert np.allclose(
                variables["params"]["query"]["bias"], treex_module.query["bias"]
            )
            assert np.allclose(
                variables["params"]["key"]["bias"], treex_module.key["bias"]
            )
            assert np.allclose(
                variables["params"]["value"]["bias"], treex_module.value["bias"]
            )
            assert np.allclose(
                variables["params"]["out"]["bias"], treex_module.out["bias"]
            )

    def test_call(self):
        key = tx.Key(42)
        inputs_q = jax.random.uniform(key, shape=(10, 20, 16))
        inputs_kv = jax.random.uniform(key, shape=(10, 20, 16))
        inputs = dict(inputs_q=inputs_q, inputs_kv=inputs_kv)

        module = tx.MultiHeadDotProductAttention(num_heads=4).init(key, inputs)

        y = module(**inputs)
        assert y.shape == (10, 20, 16)

    def test_tree(self):
        key = tx.Key(42)
        inputs_q = jax.random.uniform(key, shape=(10, 20, 16))
        inputs_kv = jax.random.uniform(key, shape=(10, 20, 16))
        inputs = dict(inputs_q=inputs_q, inputs_kv=inputs_kv)

        module = tx.MultiHeadDotProductAttention(num_heads=4).init(42, inputs)

        flat = jax.tree_leaves(module)
        assert len(flat) == 9  # q,k,v,o * 2 + rng

    def test_slice(self):
        key = tx.Key(42)
        inputs_q = jax.random.uniform(key, shape=(10, 20, 16))
        inputs_kv = jax.random.uniform(key, shape=(10, 20, 16))

        inputs = dict(inputs_q=inputs_q, inputs_kv=inputs_kv)
        module = tx.MultiHeadDotProductAttention(num_heads=4).init(42, inputs)

        flat = jax.tree_leaves(module.filter(tx.Parameter))

        assert len(flat) == 8

        flat = jax.tree_leaves(module.filter(tx.State))

        assert len(flat) == 1

    def test_jit(self):
        key = tx.Key(42)
        inputs_q = jax.random.uniform(key, shape=(10, 20, 16))
        inputs_kv = jax.random.uniform(key, shape=(10, 20, 16))

        inputs = dict(inputs_q=inputs_q, inputs_kv=inputs_kv)
        module = (
            tx.MultiHeadDotProductAttention(num_heads=4).init(key, inputs).train(False)
        )

        @jax.jit
        def f(module, **kwargs):
            return module, module(**kwargs)

        module2, y = f(module, **inputs)

        assert y.shape == (10, 20, 16)
        assert all(
            np.allclose(a, b)
            for a, b in zip(jax.tree_leaves(module), jax.tree_leaves(module2))
        )
