import jax
import jax.numpy as jnp

i = jnp.array([1, 2, 3, 4, 5])


def f(x, i):
    print("SCAN")
    return x, x


res = jax.lax.scan(f, 0, i)

print(res)
