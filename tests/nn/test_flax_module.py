import flax
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
