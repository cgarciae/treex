from typing import List, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import treex as tx


class MLP(tx.Module):
    layers: List[tx.Linear]

    def __init__(self, features: Sequence[int]):
        super().__init__()
        self.layers = [
            tx.Linear(din, dout) for din, dout in zip(features[:-1], features[1:])
        ]

    def __call__(self, x):
        for linear in self.layers[:-1]:
            x = jax.nn.relu(linear(x))
        return self.layers[-1](x)


model = MLP([1, 12, 8, 1]).init(42)

x = np.random.uniform(-1, 1, size=(100, 1))
y = 1.4 * x ** 2 - 0.3 + np.random.normal(scale=0.1, size=(100, 1))


@jax.jit
@jax.grad
def loss_fn(model, x, y):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)


# in reality use optax
def sdg(param, grad):
    return param - 0.01 * grad


# training loop
for step in range(10_000):
    grads = loss_fn(model, x, y)
    model = jax.tree_map(sdg, model, grads)

model = model.eval()
y_pred = model(x)

plt.plot(x, y, "o")
plt.plot(x, y_pred, "x")
plt.show()
