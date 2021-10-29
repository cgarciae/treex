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


# in general, try to only differentiate w.r.t. parameters
def loss_fn(params, model, x, y):
    # merge params into model
    model = model.merge(params)

    preds = model(x)
    loss = jnp.mean((preds - y) ** 2)

    # in general, the model may contain state updates
    # so it should be returned
    return loss, model


grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

# both model and optimizer are jit-able
@jax.jit
def train_step(model, x, y, optimizer):
    # select only the parameters
    params = model.parameters()

    (loss, model), grads = grad_fn(params, model, x, y)

    # update params and model
    params = optimizer.update(grads, params)
    model = model.merge(params)

    # return new model and optimizer
    return loss, model, optimizer


model = tx.Linear(1).init(42, x)
optimizer = tx.Optimizer(optax.adam(0.01)).init(model)

for step in range(300):
    loss, model, optimizer = train_step(model, x, y, optimizer)
    if step % 50 == 0:
        print(f"loss: {loss:.4f}")

model = model.eval()

X_test = np.linspace(x.min(), x.max(), 100)[:, None]
preds = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, preds, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
