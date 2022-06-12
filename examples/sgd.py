from typing import List, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import treex as tx


class MLP(tx.Module):
    layers: List[tx.Linear]

    def __init__(self, features: Sequence[int]):

        self.layers = [tx.Linear(dout) for dout in features]

    def __call__(self, x):
        for linear in self.layers[:-1]:
            x = jax.nn.relu(linear(x))
        return self.layers[-1](x)


x = np.random.uniform(-1, 1, size=(100, 1))
y = 1.4 * x**2 - 0.3 + np.random.normal(scale=0.1, size=(100, 1))

model = MLP([12, 8, 1]).init(key=42)(x)


@jax.jit
@jax.grad
def loss_fn(model, x, y):
    preds = model(x)
    return jnp.mean((preds - y) ** 2)


# in reality use optax
def sgd(param, grad):
    return param - 0.01 * grad


# training loop
for step in range(10_000):
    grads = loss_fn(model, x, y)
    model = jax.tree_map(sgd, model, grads)

model = model.eval()

X_test = np.linspace(x.min(), x.max(), 100)[:, None]
preds = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, preds, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
