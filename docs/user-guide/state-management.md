
# State Management
Treex takes a "direct" approach to state management, i.e., state is updated in-place by the Module whenever it needs to. For example, this module will calculate the running average of its input:
```python
class Average(tx.Module):
    count: jnp.ndarray = tx.State.node()
    total: jnp.ndarray = tx.State.node()

    def __init__(self):
        self.count = jnp.array(0)
        self.total = jnp.array(0.0)

    def __call__(self, x):
        self.count += np.prod(x.shape)
        self.total += jnp.sum(x)

        return self.total / self.count
```
Treex Modules that require random state will often keep a `rng` key internally and update it in-place when needed:
```python
class Dropout(tx.Module):
    key: jnp.ndarray = tx.Rng.node()

    def __init__(self, key: jnp.ndarray):
        self.key = key
        ...

    def __call__(self, x):
        key, self.key = jax.random.split(self.key)
        ...
```
Finally `


Optimizer` also performs inplace updates inside the `update` method, here is a sketch of how it works:

```python
class Optimizer(tx.Module):
    opt_state: Any = tx.OptState.node()
    optimizer: optax.GradientTransformation

    def update(self, grads, params):
        ...
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, params
        )
        ...
```
## What is the catch?
<!-- TODO: Add a list of rules to follow around jitted functions -->
State management is one of the most challenging things in JAX because of its functional nature, however here it seems effortless. What is the catch? As always there are trade-offs to consider: 

* The Pytree approach requires the user to be aware that if a Module is stateful it should propagate its state as output to jitted functions, on the other hand implementation and usage if very simple.
* Frameworks like Flax and Haiku are more explicit as to when state is updated but introduce a lot of complexity to do so.

A standard solution to this problem is: **always output the Module to merge its state**. For example, a typical loss function that contains a stateful model would look like this:

```python
@partial(jax.value_and_grad, has_aux=True)
def loss_fn(params, model, x, y):
    model = model.update(params)

    preds = model(x)
    loss = jnp.mean((preds - y) ** 2)

    return loss, model

params = model.parameters()
(loss, model), grads = loss_fn(params, model, x, y)
...
```
Here `model` is returned along with the loss through `value_and_grad` to update `model` on the outside thus persisting any changes to the state performed on the inside.

