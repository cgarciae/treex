# Treex

### Parameter Declaration
```python
class Linear(tx.Module):
    w: tx.Parameter # treex uses annotations to define structure
    b: tx.Parameter # modules contain their parameters directly

    def __init__(self, din, dout):
        # you can use an Initializer for lazy initialization
        self.w = tx.Initializer(lambda key: jax.random.uniform(key, shape=(din, dout)))
        # or just set their value directly
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b

linear = Linear(3, 5).init(42) # pass an int or PRNGKey to initialize
y = linear(x) # you can call modules directly
```

### Module Composition

```python
import treex as tx

class MLP(tx.Module):
    linear1: tx.Linear
    linear2: tx.Linear

    def __init__(din, dmid, dout):
        self.linear1 = tx.Linear(din, dmid)
        self.linear2 = tx.Linear(dmid, dout)

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        return x

mlp = MLP(3, 5, 2).init(42)
```

### Modules are Pytrees
```python
@jax.grad # no need for special versions of `jit`, `grad`, `vmap`, etc.
def loss_fn(model: MLP, x, y):  # as with any pytree Modules can be passed through `grad`
    y_pred = model(x) # just call the modules, no need for `apply`
    return jnp.mean((y_pred - y) ** 2)
```


### Slice and Merge API
The `slice` method allows you to select a subtree by filtering based on a type, all leaves that are not a subclass of such type are set to a special `Nothing` value.
```python
class MyModule(tx.Module):
    a: tx.Parameter = 1
    b: tx.State = 2
    ...

module = MyModule(...)

module.slice(tx.Parameter) # MyModule(a=1, b=Nothing)
module.slice(tx.State)     # MyModule(a=Nothing, b=2)
```
`Nothing` much like `None` is an empty pytree so it gets ignored by tree operations:

```python
jax.tree_leaves(module.Slice(tx.Parameter)) # [1]
jax.tree_leaves(module.Slice(tx.State))     # [2]
```

A typical use case is to define `params` as a `Parameter` slice and pass it as the first argument to `grad` so that the gradient is computed only that subset and immediately merge them back to the `model` before performing any computation:

```python
# we take `params` as a Parameter slice from model
# but model itself is left untouched
params = model.slice(tx.Parameter)

@jax.grad # no need for special versions of `jit`, `grad`, `vmap`, etc.
def loss_fn(params, model, x, y):
    # merge traced arrays by `grad` from `params`
    model = model.merge(params)
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
class MLP(tx.Module):
    linear1: Linear
    linear2: Linear
    ...

model = MLP(...)

model.training # True
model.linear1.training # True

model = model.train(False) # model is now in evaluation mode

model.training # False
model.linear1.training # False
```

```python
@partial(jax.jit, static_argnums=(4,))
def train_step(model, x, y, opt_state, training):
    model = model.train(training)
    ...
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

    updates, opt_state = optimizer.update(grads, opt_state, model)
    new_params = optax.apply_updates(model, updates)

    model = model.merge(new_params)

    return loss, model, opt_state


for step in range(1000):
    loss, model, opt_state = train_step(model, x, y, opt_state)
    if step % 100 == 0:
        print(f"loss: {loss:.4f}")

model = model.train(False)

X_test = np.linspace(x.min(), x.max(), 100)[:, None]
y_pred = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, y_pred, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
```