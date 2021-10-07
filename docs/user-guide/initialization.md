
# Initialization

Initialization is performed by calling the `Module.init` method or the `tx.init` function with a seed. `init` returns a new Module with all fields initialized.

There are two initialization mechanisms in Treex:

1. Using an `Initializer` object to initialize a field. 
2. Defining the `rng_init` method on a Module.

`Initializer`s contain a function that take a `key` and return an initial value, `init` replaces leaves with `Initializer` objects with the initial value their function outputs for the given key:

```python
class MyModule(tx.Module):
    a: Union[tx.Initializer, jnp.ndarray] = tx.Parameter.node()
    b: int = tx.node() # we are not setting a kind for this field for no reason

    def __init__(self):
        self.a = tx.Initializer(
            lambda key: jax.random.uniform(key, shape=(1,)))
        self.b = 2

module = MyModule() 
module # MyModule(a=Initializer, b=2)
moduel.initialized # False

module = module.init(42)  
module # MyModule(a=array([0.034...]), b=2)
module.initialized # True
```

The second is to override the `rng_init` method, this is useful for fields that require complex initialization logic.

```python
class MyModule(tx.Module):
    a: Union[jnp.ndarray, tx.Initializer] = tx.Parameter.node()
    b: Union[jnp.ndarray, None] = tx.Parameter.node()

    def __init__(self):
        self.a = tx.Initializer(
            lambda key: jax.random.uniform(key, shape=(1,)))
        self.b = None

    def rng_init(self, key):
        # self.a is already initialized at this point
        self.b = 10.0 * self.a + jax.random.normal(key, shape=(1,))

module = MyModule().init(42)
module # MyModule(a=array([0.3]), b=array([3.2]))
```
As shown here, `Initializer`s are always called before `rng_init`.
