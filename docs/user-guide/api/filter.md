# Filter

The `filter` method allows you to select a subtree by filtering based on a `kind`, all leaves whose field kind is a subclass of such type are kept, the rest are set to a special `Nothing` value.

```python
tree = MyModule(a=1, b=2)

module.filter(Parameter) # MyModule(a=1, b=Nothing)
module.filter(BatchStat) # MyModule(a=Nothing, b=2)
```

Since `Nothing` is an empty Pytree it gets ignored by tree operations, this effectively allows you to easily operate on a subset of the fields:

```python
negative = lambda x: -x

jax.tree_map(negative, module.filter(Parameter)) # MyModule(a=-1, b=Nothing)
jax.tree_map(negative, module.filter(BatchStat)) # MyModule(a=Nothing, b=-2)
```

## Shortcuts
As simple filters using the standard `TreePart` types are used often, some shortcuts are provided:

| Shortcut          | Equivalence              |
| ----------------- | ------------------------ |
| `.parameters()`   | `.filter(tx.Parameter)`  |
| `.batch_stats()`  | `.filter(tx.BatchStat)`  |
| `.rngs()`         | `.filter(tx.RNG)`        |
| `.model_states()` | `.filter(tx.ModelState)` |
| `.states()`       | `.filter(tx.State)`      |
| `.metrics()`      | `.filter(tx.Metric)`     |
| `.losses()`       | `.filter(tx.Loss)`       |
| `.logs()`         | `.filter(tx.Log)`        |

Based on this the first example can be written as:

```python
module.parameters()  # MyModule(a=1, b=Nothing)
module.batch_stats() # MyModule(a=Nothing, b=2)
```

## filter predicates
If you need to do more complex filtering, you can pass callables with the signature 

```
FieldInfo -> bool
``` 

instead of types:

```python
# all Parameters whose field name is "kernel"
module.filter(
    lambda field: issubclass(field.kind, Parameter) 
    and field.name == "kernel"
) 
# MyModule(a=Nothing, b=Nothing)
```

## multiple filters
You can some queries by using multiple filters as `*args`. For a field to be kept it will required that **all filters pass**. Since passing types by themselves are "kind filters", one of the previous examples could be written as:

```python
# all Parameters whose field name is "kernel"
module.filter(
    Parameter,
    lambda field: field.name == "kernel"
) 
# MyModule(a=Nothing, b=Nothing)
```
## inplace

If `inplace` is `True`, the input `obj` is mutated and returned. You can only update inplace if the input `obj` has a `__dict__` attribute, else a `TypeError` is raised.