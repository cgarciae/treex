from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import treex as tx

x = np.random.uniform(size=(500, 1))
y = 1.4 * x - 0.3 + np.random.normal(scale=0.1, size=(500, 1))


class Linear(tx.Module):
    w: Union[tx.Initializer, jnp.ndarray] = tx.Parameter.node()
    b: jnp.ndarray = tx.Parameter.node()

    def __init__(self, din, dout):

        self.w = tx.Initializer(lambda key: jax.random.uniform(key, shape=(din, dout)))
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b


@partial(jax.value_and_grad, has_aux=True)
def loss_fn(params, model, x, y):
    model = model.merge(params)

    y_pred = model(x)
    loss = jnp.mean((y_pred - y) ** 2)

    return loss, model


@jax.jit
def train_step(model, x, y, optimizer):
    params = model.filter(tx.Parameter)
    (loss, model), grads = loss_fn(params, model, x, y)

    # here model == params
    model = optimizer.update(grads, model)

    return loss, model, optimizer


model = Linear(1, 1).init(42)
optimizer = tx.Optimizer(optax.adam(0.01))
optimizer = optimizer.init(model)

for step in range(1000):
    loss, model, optimizer = train_step(model, x, y, optimizer)
    if step % 100 == 0:
        print(f"loss: {loss:.4f}")

model = model.eval()

X_test = np.linspace(x.min(), x.max(), 100)[:, None]
y_pred = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, y_pred, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
