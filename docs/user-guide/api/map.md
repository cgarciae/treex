
# Map

Applies a function to all leaves in the Module using `jax.tree_map`. If `filters` are given then the function will be applied only to the subset of leaves that match the filters.

For example, if we want to zero all batch stats we can do:

Example:

```python
@dataclass
class MyModule(x.Module):
    a: int = tx.Parameter.node()
    b: int = tx.BatchStat.node()

module = MyModule(a=1, b=2)

module.map(lambda _: 0, tx.BatchStat) # MyTree(a=1, b=0)
```

`map` is equivalent to `filter -> tree_map -> update` in sequence.

If `inplace` is `True`, the input `obj` is mutated and returned. You can only update inplace if the input `obj` has a `__dict__` attribute, else a `TypeError` is raised.
