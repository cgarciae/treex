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
        treex_module = tx.FlaxModule(flax_module).init(
            42,
            inputs=tx.Inputs(x, training=False),
        )

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
        rng = tx.Key(42)

        flax_module = SomeModule()
        params_key, dropout_key = tx.iter_split(tx.Key(0))
        variables = flax_module.init(
            {"params": params_key, "dropout": dropout_key},
            x,
            training,
            rng,
        )

        treex_module = tx.FlaxModule(SomeModule(), variables=variables,).init(
            42,
            inputs=tx.Inputs(x, training=False, rng=rng),
        )

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

        x = np.random.uniform(size=(2, 5, 8))
        training = True

        flax_module = SomeModule()
        params_key, dropout_key = tx.iter_split(tx.Key(0))
        variables = flax_module.init(
            {"params": params_key, "dropout": dropout_key},
            x,
            training,
        )

        treex_module = tx.FlaxModule(
            flax_module,
            variables=variables,
        ).init(42, inputs=tx.Inputs(x, training=False))
        flax_key = treex_module.next_key.key

        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(variables["batch_stats"]),
                jax.tree_leaves(treex_module.filter(tx.BatchStat)),
            )
        )

        # step 1
        next_key, flax_key = tx.iter_split(flax_key)
        y_flax, updates = flax_module.apply(
            variables,
            x,
            training,
            rngs={"dropout": next_key},
            mutable=["batch_stats"],
        )
        variables = variables.copy(updates)
        y_treex = treex_module(x, training)

        assert np.allclose(flax_key, treex_module.next_key.key)

        # step 2
        next_key, flax_key = tx.iter_split(flax_key)
        y_flax, updates = flax_module.apply(
            variables,
            x,
            training,
            rngs={"dropout": next_key},
            mutable=["batch_stats"],
        )
        variables = variables.copy(updates)
        y_treex = treex_module(x, training)
        assert np.allclose(flax_key, treex_module.next_key.key)

        assert np.allclose(y_treex, y_flax)

        # eval
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(variables["batch_stats"]),
                jax.tree_leaves(treex_module.filter(tx.BatchStat)),
            )
        )

        training = False
        y_flax = flax_module.apply(
            variables,
            x,
            training,
            mutable=False,
        )
        # variables = variables.copy(updates)
        treex_module = treex_module.eval()
        y_treex = treex_module(x, training)

        assert np.allclose(y_treex, y_flax)
