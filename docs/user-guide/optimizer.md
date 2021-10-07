
# Optimizer

Optax is an amazing library however, its optimizers are not pytrees, this means that their state and computation are separate and you cannot jit them. To solve this Treex provides a `Optimizer` class which inherits from `treeo.Tree` and can wrap any Optax optimizer. Optimizer follows a similar API as `optax.GradientTransformation` except that:

1. There is no separate `opt_state`, the Optimizer contains the state.
2. `update` by default applies the update the parameters, if you want the gradient `updates` instead you can set `apply_updates=False`.
3. `update` also updates the internal state of the Optimizer in-place.

While in Optax you would define something like this:

```python hl_lines="4 7 11"
def main():
    ...
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    ...

@partial(jax.jit, static_argnums=(4,))
def train_step(model, x, y, opt_state, optimizer): # optimizer has to be static
    ...
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    ...
    return model, loss, opt_state
```

With `tx.Optimizer` you it can be simplified to:

```python hl_lines="3 9"
def main():
    ...
    optimizer = tx.Optimizer(optax.adam(1e-3)).init(params)
    ...

jax.jit # no static_argnums needed
def train_step(model, x, y, optimizer):
    ...
    params = optimizer.update(grads, params)
    ...
    return model, loss, optimizer
```

Notice that since `tx.Optimizer` is a Pytree it was passed through `jit` naturally without the need to specify `static_argnums`.
