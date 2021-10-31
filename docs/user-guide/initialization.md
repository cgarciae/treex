
# Initialization

Initialization is performed by calling the `Module.init` method, `init` returns a new Module with all fields initialized.

There are three initialization mechanisms for Modules in Treex:

1. Using `Module.initializing` inside `__call__`. 
2. Using a field `Initializer` object.
3. Defining the `rng_init` method.


### Module.initializing
During the forward pass you can check if the Module is initialized by calling `self.initializing()` and assign the fields that need to be initialized then and there. If you need access to a RNG key, you can use `tx.next_key()` inside a `self.initializing()` block ONLY, this will use the key passed during `init`.

```python
def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    if self.initializing():
        self.w = jax.random.uniform(
            key=tx.next_key(), 
            shape=[x.shape[-1], self.features_out]
        )
```
The benefit of this approach is that you can leverage the shape of the input to initialize the parameters. This method is used by most core layers provided by Treex.

### Field Initializer
`Initializer`s contain a function that take a `key` and return an initial value, `init` will replace leaves with `Initializer` objects with the initial value their function outputs for the given key:

```python
class MyModule(tx.Module):
    a: Union[tx.Initializer, jnp.ndarray] = tx.Parameter.node()

    def __init__(self):
        self.a = tx.Initializer(
            lambda key: jax.random.uniform(key, shape=(1,))
        )

module = MyModule().init(42)  
# > MyModule(a=array([0.034...]))
```
This method is use for fields who's intialization doesn't require shape inference and doesn't depend on information of other fields.

### rng_init
If you Module doesn't require shape inference but `Initializer` is not enough, you can override the `rng_init` method.

```python
class MyModule(tx.Module):
    a: Optional[jnp.ndarray] = tx.Parameter.node()
    b: Optional[jnp.ndarray] = tx.Parameter.node()

    def __init__(self):
        self.a = None 
        self.b = None

    def rng_init(self):
        self.a = jax.random.uniform(tx.next_key(), shape=(1,)))
        self.b = 10.0 * self.a + jax.random.normal(key, shape=(1,))

module = MyModule().init(42)
module # MyModule(a=array([0.3...]), b=array([3.2...]))
```

### Intialization order
The order of initialization is:

1. First all field `Initializers` are called.
2. Second all `rng_init` methods are called.
3. Lastly the `__call__` method is called.