# Treex

_A simple pure PyTree Module system for JAX_

* **Simple and Intuitive**: Modules are simple Python objects that respect Object Oriented semantics and should make PyTorch users feel at home, no need for separate dictionary structures or complex `apply` methods.
* **PyTree-based**: Modules are registered as JAX PyTrees so they can be used with any JAX function, no need for special versions of `jit`, `grad`, `vmap`, etc.
* **Expressive**: By using type annotations you can tell Treex what the different parts of your module do, this leads to a very powerful state management solution.
* **Doesn't reinvent the wheel**: Writting high-quality, battle-tested code for common layers is hard, so currently Modules in `treex.nn` are wrappers over their **Flax** counterparts, they keep the same signatures so Flax users feel at home but still grant them the simple Pytorch-like behavior Treex brings.

## Why Treex?
Despite all JAX benefits, current Module systems are not intuitive to new users and add additional complexity not present in frameworks like PyTorch or Keras. Treex takes insparation from S4TF and delivers an intuitive experience using JAX PyTree infrastructure.

<details>
<summary>Current Alternative's Drawbacks and Solutions</summary>

Currently we have many alternatives like Flax, Haiku, Objax, that have one or more of the following drawbacks:

* Module structure and parameter structure are separate, parameters have to be manipulated around by the user which is not intuitive. In Treex, parameters are stored in the modules themselves and can be accessed directly.
* Monadic architecture's add complexity, both Flax and Haiku use an `apply` method to call modules which sets a context with parameters, rng, etc, which add an additional overhead to the API and creates an asymmetry to how Modules are used inside and outside a context. In Treex you can just call the modules directly.
* Parameter surgery is very difficult to implement, if you want to transfer a pretrained module or submodule as part of a new module, you have to know precisely how to extract their parameters and how to insert them into the new parameter structure / dictionaries such that it is in agreement with the new module structure. In Treex, just as in PyTorch / Keras you just pass the (sub)module to the new module and parameters go with them.
* Deviate from JAX semantics and require special versions of `jit`, `grad`, `vmap`, etc, which makes it harder to integrate with other JAX libraries. Treex's Modules are plain old JAX PyTrees and are compatible with any JAX library that supports them.
* Other PyTree-based approaches like Parallax and Equinox don't have a full state management solution to handle complex state as you see in Flax. Treex has the Filter and Update API that is very expressive and can effectively handle systems with complex state.

</details>

## Installation
Install using pip:
```bash
pip install treex
```

## Status
Treex is currently in **alpha** stage, however, its internal implementation is very simple so its probably near completion.

Current roadmap:
- [x] Finish prototyping the API
- [ ] Finalize basic API
- [ ] Wrap all Flax Linen Modules
- [ ] Document public API
- [ ] Create documentation site

Since Treex is not a Google-related project its success will depend largely on support from the community.

## Getting Started
This is a small appetizer to give you a feel for how using Treex looks like, be sure to checkout the [Guide section](#guide) below for details on more advanced usage.
```python
from typing import Sequence, List

import jax
import jax.numpy as jnp
import numpy as np
import treex as tx


class MLP(tx.Module):
    layers: List[tx.Linear]

    def __init__(self, features: Sequence[int]):
        self.layers = [
            tx.Linear(din, dout) 
            for din, dout in zip(features[:-1], features[1:])
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
```

## Guide
### Defining Modules
Treex Modules have the following characteristics:
* They inherit from `tx.Module`.
* Fields for parameter and submodule **MUST** be marked using a _valid_ type annotation.


```python
class Linear(tx.Module):
    w: tx.Parameter
    b: tx.Parameter

    def __init__(self, din, dout):
        self.w = tx.Initializer(
            lambda key: jax.random.uniform(key, shape=(din, dout)))
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b

linear = Linear(3, 5).init(42)
y = linear(x)
```

Valid type annotations include:
* Subtypes of `tx.TreePart` e.g. `tx.Parameter`, `tx.BatchStat`, etc.
* Subtypes of `tx.Module` e.g. `tx.Linear`, custom Module types, etc.
* Generic subtypes from the `typing` module of the previous e.g. `List[tx.Parameter]` or `Dict[str, tx.Linear]`.

Type annotations that do not comform to the above rules will be ignored and the field will not be counted as part of the PyTree.

```python
class MLP(tx.Module):
    layers: List[tx.Linear]

    def __init__(self, features: Sequence[int]):
        self.layers = [
            tx.Linear(din, dout) 
            for din, dout in zip(features[:-1], features[1:])
        ]

    def __call__(self, x):
        for linear in self.layers[:-1]:
            x = jax.nn.relu(linear(x))
        return self.layers[-1](x)

mlp = MLP([3, 5, 2]).init(42)
```

### Pytrees
Since Modules are pytrees they can be arguments to JAX functions such as `jit`, `grad`, `vmap`, etc, and the `jax.tree_*` function family.
```python
@jax.jit
@jax.grad
def loss_fn(model, x, y):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)

def sdg(param, grad):
    return param - 0.01 * grad

model = MLP(...).init(42)

grads = loss_fn(model, x, y)
model = jax.tree_map(sdg, model, grads)
```
This makes Treex Modules compatible with tooling from the JAX ecosystem, and enables correct unification of Modules as both the parameter containers and the definition of the foward computation.

### Initialization

```python
class MyModule(tx.Module):
    a: tx.Parameter
    b: tx.Parameter

    def __init__(self):
        self.a = tx.Initializer(lambda key: 1)
        self.b = 2

module = MyModule() 
module # MyModule(a=Initializer, b=2)
moduel.initialized # False

module = module.init(42)  
module # MyModule(a=1, b=2)
module.initialized # True
```

```python
class MyModule(tx.Module):
    a: tx.Parameter
    b: tx.Parameter

    def __init__(self):
        self.a = tx.Initializer(lambda key: 1)
        self.b = 2

module = MyModule() 
module # MyModule(a=Initializer, b=2)
moduel.initialized # False

module = module.init(42)  
module # MyModule(a=1, b=2)
module.initialized # True
```



### Filter and Update API
The `filter` method allows you to select a subtree by filtering based on a type, all leaves that are not a subclass of such type are set to a special `Nothing` value.
```python
class MyModule(tx.Module):
    a: tx.Parameter = np.array(1)
    b: tx.BatchStat = np.array(2)
    ...

module = MyModule(...)

module.filter(tx.Parameter) # MyModule(a=array([1]), b=Nothing)
module.filter(tx.BatchStat) # MyModule(a=Nothing, b=array([2]))
```
`Nothing` much like `None` is an empty pytree so it gets ignored by tree operations:

```python
jax.tree_leaves(module.filter(tx.Parameter)) # [array([1])]
jax.tree_leaves(module.filter(tx.BatchStat)) # [array([2])]
```

A typical use case is to define `params` as a `Parameter` filter and pass it as the first argument to `grad` so that the gradient is computed only that subset and immediately update them back to the `model` before performing any computation:

```python
# we take `params` as a Parameter filter from model
# but model itself is left untouched
params = model.filter(tx.Parameter)

@jax.grad 
def loss_fn(params, model, x, y):
    # update traced arrays by `grad` from `params`
    model = model.update(params)
    ...

grads = loss_fn(params, model, x, y) 

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params) # only needs params
```

### State Management
```python
class Average(tx.Module):
    count: tx.State
    total: tx.State

    def __init__(self):
        self.count = jnp.array(0)
        self.total = jnp.array(0.0)

    def __call__(self, x):
        self.count += np.prod(x.shape)
        self.total += jnp.sum(x)

        return self.total / self.count
```

```python
class Dropout(tx.Module):
    rng: tx.Rng

    def __init__(self, rate: float):
        self.rate = rate
        self.rng = tx.Initializer(lambda key: key) # its just a PRNGKey

    def __call__(self, x):
        # RNG is just State, update in place as well
        key, self.rng = jax.random.split(self.rng)
        mask = jax.random.bernoulli(key, self.rate, x.shape)
        ...
```


### Training State
```python
x = np.random.randn(10, 3)
model = tx.Dropout(0.5).init(42)

y1 = model(x)
y2 = model(x)

model.training       # True
np.allcloase(y1, y2) # False

# deterministic in eval mode
model = model.eval()

y1 = model(x)
y2 = model(x)

model.training      # False
np.allclose(y1, y2) # True
```

### Parameter Surgery
```python
class VAE(tx.Module):
    encoder: Encoder
    decoder: Decoder
    ...

vae = VAE(...)

# train VAE
...

# extract decoder to generate samples
decoder = vae.decoder
samples = decoder(z)
```

### Custom Annotations

### Container Types

### Full Example

```python
from functools import partial
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import treex as tx

x = np.random.uniform(size=(500, 1))
y = 1.4 * x - 0.3 + np.random.normal(scale=0.1, size=(500, 1))

# treex already defines tx.Linear but we can define our own
class Linear(tx.Module):
    w: tx.Parameter
    b: tx.Parameter

    def __init__(self, din, dout):
        self.w = tx.Initializer(lambda key: jax.random.uniform(key, shape=(din, dout)))
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b


model = Linear(1, 1).init(42)
optimizer = optax.adam(0.01)

opt_state = optimizer.init(model.filter(tx.Parameter))


@partial(jax.value_and_grad, has_aux=True)
def loss_fn(params, model, x, y):
    model = model.update(params)

    y_pred = model(x)
    loss = jnp.mean((y_pred - y) ** 2)

    return loss, model


@jax.jit
def train_step(model, x, y, opt_state):
    params = model.filter(tx.Parameter)
    (loss, model), grads = loss_fn(params, model, x, y)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    new_params = optax.apply_updates(params, updates)

    model = model.update(new_params)

    return loss, model, opt_state


for step in range(1000):
    loss, model, opt_state = train_step(model, x, y, opt_state)
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