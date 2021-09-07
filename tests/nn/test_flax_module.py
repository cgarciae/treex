import flax
import jax
import numpy as np

import treex as tx


class TestFlaxModule:
    def test_flax_module(self):
        class SomeModule(flax.linen.Module):
            @flax.linen.compact
            def __call__(self, x, training):
                x = flax.linen.Dense(16)(x)
                x = flax.linen.BatchNorm()(x, use_running_average=not training)
                x = flax.linen.Dropout(0.5)(x, deterministic=not training)
                x = flax.linen.Conv(16, [3])(x)

                return x

        x = np.ones((2, 5, 8), dtype=np.float32)
        flax_module = SomeModule()
        treex_module = tx.FlaxModule(
            flax_module,
            sample_inputs=tx.Inputs(x, training=True),
        ).init(42)

        y = treex_module(x, training=True)

    def test_pretrained_flax_module(self):
        class SomeModule(flax.linen.Module):
            @flax.linen.compact
            def __call__(self, x, training, rng):
                x = flax.linen.Dense(16)(x)
                x = flax.linen.BatchNorm()(x, use_running_average=not training)
                x = flax.linen.Dropout(0.5)(x, deterministic=not training, rng=rng)
                x = flax.linen.Conv(16, [3])(x)

                return x

        x = np.ones((2, 5, 8), dtype=np.float32)
        training = True
        rng = jax.random.PRNGKey(42)

        flax_module = SomeModule()
        params_key, dropout_key = jax.random.split(jax.random.PRNGKey(0))
        variables = flax_module.init(
            {"params": params_key, "dropout": dropout_key},
            x,
            training,
            rng,
        )

        treex_module = tx.FlaxModule(
            SomeModule(),
            sample_inputs=tx.Inputs(x, training=True),
            variables=variables,
        ).init(42)

        y_treex = treex_module(x, training, rng)
        y_treex = treex_module(x, training, rng)

        y_flax, updates = flax_module.apply(
            variables,
            x,
            training,
            rng,
            mutable=["batch_stats"],
        )
        variables = variables.copy(updates)
        y_flax, updates = flax_module.apply(
            variables,
            x,
            training,
            rng,
            mutable=["batch_stats"],
        )
        variables = variables.copy(updates)

        assert np.allclose(y_treex, y_flax)

        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(variables["params"]),
                jax.tree_leaves(treex_module.filter(tx.Parameter)),
            )
        )

    def test_pretrained_flax_module_no_rng(self):
        class SomeModule(flax.linen.Module):
            @flax.linen.compact
            def __call__(self, x, training):
                x = flax.linen.Dense(16)(x)
                x = flax.linen.BatchNorm()(x, use_running_average=not training)
                x = flax.linen.Dropout(0.5)(x, deterministic=not training)
                x = flax.linen.Conv(16, [3])(x)

                return x

        x = np.ones((2, 5, 8), dtype=np.float32)
        training = True

        flax_module = SomeModule()
        params_key, dropout_key = jax.random.split(jax.random.PRNGKey(0))
        variables = flax_module.init(
            {"params": params_key, "dropout": dropout_key},
            x,
            training,
        )

        treex_module = tx.FlaxModule(
            flax_module,
            sample_inputs=tx.Inputs(x, training=True),
            variables=variables,
        ).init(42)
        flax_key = treex_module.rng_seq.key

        # step 1
        next_key, flax_key = jax.random.split(flax_key)
        y_flax, updates = flax_module.apply(
            variables,
            x,
            training,
            rngs={"dropout": next_key},
            mutable=["batch_stats"],
        )
        variables = variables.copy(updates)
        y_treex = treex_module(x, training)

        assert np.allclose(flax_key, treex_module.rng_seq.key)

        # step 2
        next_key, flax_key = jax.random.split(flax_key)
        y_flax, updates = flax_module.apply(
            variables,
            x,
            training,
            rngs={"dropout": next_key},
            mutable=["batch_stats"],
        )
        variables = variables.copy(updates)
        y_treex = treex_module(x, training)
        assert np.allclose(flax_key, treex_module.rng_seq.key)

        assert np.allclose(y_treex, y_flax)
