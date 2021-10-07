from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import treex as tx
from treex.nn import dropout


class MLP(tx.Module):
    linear1: tx.Linear
    dropout1: tx.Dropout
    linear2: tx.Linear

    def __init__(self, din, dmid, dout, dropout: float = 0.5):

        self.linear1 = tx.Linear(din, dmid)
        self.dropout1 = tx.Dropout(dropout)
        self.linear2 = tx.Linear(dmid, dout)

    def __call__(self, x):
        x = jax.nn.relu(self.dropout1(self.linear1(x)))
        x = self.linear2(x)
        return x


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

    new_params = optimizer.update(grads, params)
    model = model.merge(new_params)

    return loss, model, optimizer


np.random.seed(69)
x = np.random.uniform(-1, 1, size=(500, 1))
y = 1.4 * x ** 2 - 0.3 + np.random.normal(scale=0.1, size=(500, 1))

model = MLP(1, 32, 1, dropout=0.1).init(42)
optimizer = tx.Optimizer(optax.adam(0.001))
optimizer = optimizer.init(model.filter(tx.Parameter))

for step in range(20_000):
    idx = np.random.choice(len(x), size=64, replace=False)
    loss, model, optimizer = train_step(model, x[idx], y[idx], optimizer)
    if step % 2000 == 0:
        print(f"[{step}] loss: {loss:.4f}")

model = model.eval()

X_test = np.linspace(x.min(), x.max(), 100)[:, None]
y_pred = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, y_pred, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
