import jax
import jax.numpy as jnp


def all_pairs(f):
    f = jax.vmap(f, in_axes=(None, 0))
    f = jax.vmap(f, in_axes=(0, None))
    return f


def distance(a, b):
    return jnp.linalg.norm(a - b)


distances = all_pairs(distance)

A = jnp.array([[0, 0], [1, 1], [2, 2]])
B = jnp.array([[-10, -10], [-20, -20]])

D = distances(A, B)

print(D)
