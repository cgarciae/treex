<!-- ## Defining Modules -->

### Basic Modules
Modules in Treex usually follow this recipe:

* They inherit from `tx.Module`.
* Parameter-like fields are declared with a `tx.TreePart` subclass kind e.g. `tx.Parameter.node()`
* Hyper-parameters fields usually don't contain a declaration so they are static.
* Modules can be defined as dataclasses or regular classes without any limitations.
* While not mandatory, they usually perform shape inference.

For example, a basic Module will tend to look like this:

```python
import treex as tx

class Linear(tx.Module):
    # use Treeo's API to define Parameter nodes
    w: jnp.ndarray = tx.Parameter.node()
    b: jnp.ndarray = tx.Parameter.node()

    def __init__(self, features_out: int):
        self.features_out = features_out

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # init will call forward, we can know if we are inside it
        if self.initializing():
            # `next_key` only available during `init`
            key = tx.next_key() 
            # leverage shape inference
            self.w = jax.random.uniform(
                key, shape=[x.shape[-1], self.features_out]
            )
            self.b = jnp.zeros(shape=[self.features_out])

        # linear forward
        return jnp.dot(x, self.w) + self.b

model = Linear(10).init(key=42, inputs=x)   
```
### Composite Modules

Composite modules have the following characteristics:

* Their submodule fields are usually not declared, they are usually detected by their runtime value.
* Submodules are either created during `__init__` or directly in `__call__` when using `@compact`.

```python
class MLP(tx.Module):
    def __init__(self, features: Sequence[int]):
        self.features = features

    @tx.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for units in self.features[:-1]:
            x = Linear(units)(x)
            x = jax.nn.relu(x)

        return Linear(self.features[-1])(x)

model = MLP([32, 10]).init(key=42, inputs=x)
```
If you don't want to use compact, you can create a list of `Linear` modules during `__init__` and use them in `__call__`. While in Pytorch you would create a `ModuleList` or `ModuleDict` to do this, in Treex you just need to use a (possibly generic) type annotation on the class field that contains a Module type (e.g. `Linear`).

```python
class MLP(tx.Module):
    layers: List[Linear] # mandatory: registers field as a node

    def __init__(self, features: Sequence[int]):
        self.layers = [Linear(units) for units in features]

    @tx.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = jax.nn.relu(x)

        return self.layers[-1](x)

model = MLP([32, 10]).init(key=42, inputs=x)
```
For more information check out Treeo's [Node Policy](https://cgarciae.github.io/treeo/user-guide/node-policy).