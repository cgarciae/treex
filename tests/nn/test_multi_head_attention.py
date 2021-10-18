import jax
import jax.numpy as jnp

import treex as tx


class TestMultiHeadAttention:
    def test_connects(self):

        batch_size = 5
        input_size = 8
        output_size = 16
        num_heads = 4
        num_queries = 10
        num_keys = 12

        module = tx.MultiHeadAttention(
            input_size, num_heads, output_size=output_size
        ).init(42)

        key = jax.random.PRNGKey(0)
        queries = jax.random.uniform(key, (batch_size, num_queries, input_size))
        keys = jax.random.uniform(key, (batch_size, num_keys, input_size))

        y = module(queries, keys)
        assert y.shape == (batch_size, num_queries, output_size)

        y, coefs = module(queries, keys, return_attn_coef=True)
        assert y.shape == (batch_size, num_queries, output_size)
        assert coefs.shape == (batch_size, num_heads, num_queries, num_keys)

    def test_connects_mask(self):

        batch_size = 5
        input_size = 8
        output_size = 16
        num_heads = 4
        num_queries = 10
        num_keys = 12

        module = tx.MultiHeadAttention(
            input_size, num_heads, output_size=output_size
        ).init(42)

        key = jax.random.PRNGKey(0)
        queries = jax.random.uniform(key, (batch_size, num_queries, input_size))
        keys = jax.random.uniform(key, (batch_size, num_keys, input_size))

        # single mask
        mask = jax.random.uniform(key, (num_queries, num_keys))

        y = module(queries, keys, mask=mask)
        assert y.shape == (batch_size, num_queries, output_size)

        y, coefs = module(queries, keys, return_attn_coef=True, mask=mask)
        assert y.shape == (batch_size, num_queries, output_size)
        assert coefs.shape == (batch_size, num_heads, num_queries, num_keys)

        # mask per head
        mask = jax.random.uniform(key, (num_heads, num_queries, num_keys))

        y = module(queries, keys, mask=mask)
        assert y.shape == (batch_size, num_queries, output_size)

        y, coefs = module(queries, keys, return_attn_coef=True, mask=mask)
        assert y.shape == (batch_size, num_queries, output_size)
        assert coefs.shape == (batch_size, num_heads, num_queries, num_keys)
