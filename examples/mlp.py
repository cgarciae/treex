from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import treex as tx
from treex.nn import dropout


class MLP(tx.Module):
    linear: tx.Linear
    dropout: tx.Dropout
    linear2: tx.Linear

    def __init__(self, dmid, dout, dropout: float = 0.5):

        self.dmid = dmid
        self.dout = dout
        self.dropout_rate = dropout

    @tx.compact
    def __call__(self, x):
        x = tx.Linear(self.dmid)(x)
        x = tx.Dropout(self.dropout_rate)(x)
        x = jax.nn.relu(x)
        x = tx.Linear(self.dout)(x)
        return x


Model = MLP


def loss_fn(params: Model, key: jnp.ndarray, model: Model, x, y):
    model = model.merge(params)

    preds, model = model.apply(key, x)

    loss = jnp.mean((preds - y) ** 2)

    return loss, model


@jax.jit
def train_step(
    key: jnp.ndarray,
    model: Model,
    optimizer: tx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    params = model.trainable_parameters()
    loss_key, key = jax.random.split(key)

    (loss, model), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, key, model, x, y
    )

    params, optimizer = optimizer.update(grads, params)
    model = model.merge(params)

    return loss, key, model, optimizer


np.random.seed(69)
x = np.random.uniform(-1, 1, size=(500, 1))
y = 1.4 * x**2 - 0.3 + np.random.normal(scale=0.1, size=(500, 1))

model = MLP(32, 1, dropout=0.1).init(42, x)
optimizer = tx.Optimizer(optax.adam(0.001)).init(model.trainable_parameters())
key = tx.Key(42)

for step in range(20_000):
    idx = np.random.choice(len(x), size=64, replace=False)
    loss, key, model, optimizer = train_step(key, model, optimizer, x[idx], y[idx])
    if step % 2000 == 0:
        print(f"[{step}] loss: {loss:.4f}")

model = model.eval()

X_test = np.linspace(x.min(), x.max(), 100)[:, None]
preds = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, preds, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
