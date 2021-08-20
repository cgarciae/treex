from functools import partial
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import treex as tx

np.random.seed(69)

x = np.random.uniform(-1, 1, size=(500, 1))
y = 1.4 * x ** 2 - 0.3 + np.random.normal(scale=0.1, size=(500, 1))


class MLP(tx.Module):
    linear1: tx.Linear
    linear2: tx.Linear

    def __init__(self, din, dmid, dout):
        self.linear1 = tx.Linear(din, dmid)
        self.batch_norm1 = tx.BatchNorm(dmid)
        self.linear2 = tx.Linear(dmid, dout)

    def __call__(self, x):
        x = tx.sequence(
            self.linear1,
            # self.batch_norm1,
            jax.nn.relu,
        )(x)
        x = self.linear2(x)
        return x


model = MLP(1, 32, 1).init(42)
optimizer = optax.adam(0.00005)

opt_state = optimizer.init(model.slice(tx.Parameter))


@partial(jax.value_and_grad, has_aux=True)
def loss_fn(params, model, x, y):
    model = model.merge(params)
    y_pred = model(x)
    loss = jnp.mean((y_pred - y) ** 2)

    return loss, model


@jax.jit
def train_step(model, x, y, opt_state):
    params = model.slice(tx.Parameter)
    (loss, model), grads = loss_fn(params, model, x, y)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    model = model.merge(params)

    return loss, model, opt_state


for step in range(20000):
    idx = np.random.choice(len(x), size=64, replace=False)
    loss, model, opt_state = train_step(model, x[idx], y[idx], opt_state)
    if step % 500 == 0:
        print(f"loss: {loss:.4f}")

model = model.train(False)

X_test = np.linspace(x.min(), x.max(), 100)[:, None]
y_pred = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, y_pred, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
