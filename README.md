# Treex

_A Pytree Module system for Deep Learning in JAX_

* **Intuitive**: Modules are simple Python objects that respect Object-Oriented semantics and should make PyTorch users feel at home, with no need for separate dictionary structures or complex `apply` methods.
* **Pytree-based**:  Modules are registered as JAX PyTrees, enabling their use with any JAX function. No need for specialized versions of `jit`, `grad`, `vmap`, etc.
* **Expressive**: In Treex you use type annotations to define what the different parts of your module represent (submodules, parameters, batch statistics, etc), this leads to a very flexible and powerful state management solution.
* **Flax-based Implementations**: Writing high-quality, battle-tested code for common layers is hard. For this reason Modules in `treex.nn` are wrappers over their Flax counterparts. We keep identical signatures, enabling Flax users to feel at home but still benefiting from the simpler Pytorch-like experience Treex brings.

Treex is implemented on top of [Treeo](https://github.com/cgarciae/treeo), Treex vendors all of Treeo's public API for ease of usage.

[Documentation](https://cgarciae.github.io/treex) | [User Guide](https://cgarciae.github.io/treex/user-guide/intro)

## Why Treex?
Despite all JAX benefits, current Module systems are not intuitive to new users and add additional complexity not present in frameworks like PyTorch or Keras. Treex takes inspiration from S4TF and delivers an intuitive experience using JAX Pytree infrastructure.

<details>
<summary>Current Alternative's Drawbacks and Solutions</summary>

Currently we have many alternatives like Flax, Haiku, Objax, that have one or more of the following drawbacks:

* Module structure and parameter structure are separate, and parameters have to be manipulated around by the end-user, which is not intuitive. In Treex, parameters are stored in the modules themselves and can be accessed directly.
* Monadic architecture adds complexity. Flax and Haiku use an `apply` method to call modules that set a context with parameters, rng, and different metadata, which adds additional overhead to the API and creates an asymmetry in how Modules are being used inside and outside a context. In Treex, modules can be called directly.
* Among different frameworks, parameter surgery requires special consideration and is challenging to implement. Consider a standard workflow such as transfer learning, transferring parameters and state from a  pre-trained module or submodule as part of a new module; in different frameworks, we have to know precisely how to extract their parameters and how to insert them into the new parameter structure/dictionaries such that it is in agreement with the new module structure. In Treex, just as in PyTorch / Keras, we enable to pass the (sub)module to the new module, and parameters are automatically added to the new structure.
* Multiple frameworks deviate from JAX semantics and require particular versions of `jit`, `grad`, `vmap`, etc., which makes it harder to integrate with other JAX libraries. Treex's Modules are plain old JAX PyTrees and are compatible with any JAX library that supports them.
* Other Pytree-based approaches like Parallax and Equinox do not have a total state management solution to handle complex states as encountered in Flax. Treex has the Filter and Update API, which is very expressive and can effectively handle systems with a complex state.

</details>

## Installation
Install using pip:
```bash
pip install treex
```

## Status
Treex is in an early stage, things might brake between versions but we will respect semanting versioning. While more testing is needed, since Treex layers are numerically equivalent to Flax this borrows some maturity and yields more confidence over its results. Feedback is much appreciated.

**Roadmap**:

- [x] Finish prototyping core API
- [ ] Wrap all Flax Linen Modules
- [x] Document public API
- [x] Create documentation site


## Getting Started
<!-- Remake Getting Started now that most content is in the User Guide -->

This is a small appetizer to give you a feel for how using Treex looks like, be sure to checkout the [Guide section](#guide) below for details on more advanced usage.
```python
from typing import Sequence, List

import jax
import jax.numpy as jnp
import numpy as np
import treex as tx

# you can use tx.MLP but we will create our own as an example
class MLP(tx.Module):
    layers: List[tx.Linear] = tx.node()

    def __init__(self, features: Sequence[int]):
        self.layers = [
            tx.Linear(din, dout) 
            for din, dout in zip(features[:-1], features[1:])
        ]

    def __call__(self, x):
        for linear in self.layers[:-1]:
            x = jax.nn.relu(linear(x))
        return self.layers[-1](x)

@jax.jit
@jax.grad
def loss_fn(model, x, y):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)

# in reality use optax
def sdg(param, grad):
    return param - 0.01 * grad

model = MLP([1, 12, 8, 1]).init(42)

x = np.random.uniform(-1, 1, size=(100, 1))
y = 1.4 * x ** 2 - 0.3 + np.random.normal(scale=0.1, size=(100, 1))

# training loop
for step in range(10_000):
    grads = loss_fn(model, x, y)
    model = jax.tree_map(sdg, model, grads)

model = model.eval()
y_pred = model(x)
```

#### Stateful Module example
Here is an example of creating a stateful module of a `RollingMean` metric and using them with `jax.jit`:

```python
class RollingMean(tx.Module):
    count: jnp.ndarray = tx.State.node()
    total: jnp.ndarray = tx.State.node()

    def __init__(self):
        self.count = jnp.array(0)
        self.total = jnp.array(0.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        self.count += np.prod(x.shape)
        self.total += x.sum()

        return self.total / self.count

@jax.jit
def update(x: jnp.ndarray, metric: RollingMean) -> Tuple[jnp.ndarray, RollingMean]:
    mean = metric(x)
    return mean, metric # return mean value and updated metric


metric = RollingMean()

for i in range(10):
    x = np.random.uniform(-1, 1, size=(100, 1))
    mean, metric = update(x, metric)

print(mean)
```

#### Linear Regression from scratch example

```python
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

```
