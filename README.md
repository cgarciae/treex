# Treex

### Simple Python objects
```python
class Linear(tx.Module): # Module class is very simple, doesn't use MetaClass magic
    w: tx.Parameter # just use simple annotations to define the structure
    b: tx.Parameter 

    def __init__(self, din, dout):
        # modules contain their parameters directly!
        # use an Initializer for lazy initialization
        self.w = tx.Initializer(lambda key: jax.random.uniform(key, shape=(din, dout)))
        # or just set their value directly
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b

linear = Linear(3, 5).init(42) # initialization is super easy
y = linear(x) # you can call modules directly
```

### Pytorch-like APIs

```python
import treex as tx

class MLP(tx.Module):
    linear1: Linear
    linear2: Linear

    def __init__(din, dmid, dout):
        self.linear1 = Linear(din, dmid)
        self.linear2 = Linear(dmid, dout)

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        return x
```

### Modules are Pytrees
```python
@jax.grad # no need for special versions of `jit`, `grad`, `vmap`, etc.
def loss_fn(model: MLP, x, y):  # as with any pytree Modules can be passed through `grad`
    y_pred = model(x) # just call the modules, no need for `apply`
    return jnp.mean((y_pred - y) ** 2)
```

### Simple State Management
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

### Slice and Merge API
```python
params: MLP = model.slice(tx.Parameter)
states: MLP = model.slice(tx.State)

@partial(jax.value_and_grad) # no need for special versions of `jit`, `grad`, `vmap`, etc.
def loss_fn(params, states, x, y):  # will only differentiate w.r.t. params
    model = params.merge(states) # merge parameters and states back into the complete model
    ...

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params) # only params from the optimizer
```

### Simple "parameter surgery"
```python
class VAE(tx.Module):
    encoder: Encoder
    decoder: Decoder
    ...

vae = VAE(...)

# train VAE....

# extract decoder to generate samples
decoder = vae.decoder
samples = decoder(z)
```


### Full Example

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import treex as tx

x = np.random.uniform(size=(500, 1))
y = 1.4 * x - 0.3 + np.random.normal(scale=0.1, size=(500, 1))


class Linear(tx.Module):
    w: tx.Parameter
    b: tx.Parameter

    def __init__(self, din, dout):
        self.w = tx.Initializer(lambda key: jax.random.uniform(key, shape=(din, dout)))
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b


model = Linear(1, 1).init(42)
optimizer = optax.adam(0.001)

opt_state = optimizer.init(model)


@jax.value_and_grad
def loss_fn(model, x, y):
    pred = model(x)
    return jnp.mean((pred - y) ** 2)


@jax.jit
def train_step(model, x, y, opt_state):
    loss, grads = loss_fn(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = optax.apply_updates(model, updates)

    return loss, model, opt_state


for step in range(1000):
    loss, model, opt_state = train_step(model, x, y, opt_state)
    if step % 100 == 0:
        print(f"loss: {loss:.4f}")


X_test = np.linspace(x.min(), x.max(), 100)[:, None]
y_pred = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, y_pred, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
```