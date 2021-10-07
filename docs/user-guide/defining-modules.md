<!-- ## Defining Modules -->

Modules in Treex usually follow this recipe:

* They inherit from `tx.Module`.
* Parameter-like fields are declared with a `tx.TreePart` subclass kind e.g. `tx.Parameter.node()`
* Hyper-parameters fields usually don't contain a declaration so they are static.
* Submodules fields are usually not declared, instead they are infered by the type annotations or runtime values.
* Modules can be defined as dataclasses or regular classes without any limitations.

For example, a basic Module will tend to look like this:

```python
class Linear(tx.Module):
    # din: int # annotation not needed, inferred as static
    w: Union[tx.Initializer, jnp.ndarray] = tx.Parameter.node() # node field
    b: jnp.ndarray = tx.Parameter.node() # node field

    def __init__(self, din, dout):
        self.w = tx.Initializer(
            lambda key: jax.random.uniform(key, shape=(din, dout))
        )
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b

linear = Linear(3, 5).init(42)
y = linear(x)
```
While composite Module will tend to look like this:

```python
class MLP(tx.Module):
    # features: Sequence[int], annotation not needed, infered as static
    layers: List[tx.Linear] # mandatory annotation, infered as node because Modules are treeo.Trees

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
