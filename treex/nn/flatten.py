import einops
import jax.numpy as jnp

from treex.module import Module


class Flatten(Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return einops.rearrange(x, "batch ... -> batch (...)")
