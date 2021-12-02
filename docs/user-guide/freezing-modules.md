
# Freezing Modules

`Module`s have a `.frozen` property that specifies whether the module is frozen or not, Modules such as `Dropout` and `BatchNorm` which will behave differently based on its value. To switch between modes, use the `.freeze()` and `.unfreeze()` methods, they return a new Module whose `frozen` state and the state of all of its submodules (recursively) are set to the desired value.

```python hl_lines="16"
class ConvBlock(tx.Module):
    ...

model = tx.Sequential(
    ConvBlock(3, 32),
    ConvBlock(32, 64),
    ConvBlock(64, 128),
    ...
)

# train model
...

# freeze some layers
for layer in model.layers[:-1]:
    layer.freeze(inplace=True)

# fine-tune the model
...
```
In this example we can leveraged the fact that `Sequential` has its submodules in `.layers` to freeze all but the last layers.

Freezing modules is useful for tasks such as Transfer Learning where you want to keep most of the weights in a model unchange and train only a few of them on a new dataset. If you have a backbone you can just freeze the entire model.

```python hl_lines="2 16"
backbone = get_pretrained_model()
backbone = backbone.freeze()

model = tx.Sequential(
    backbone,
    tx.Linear(backbone.output_features, 10)
).init(42)

...
# Initialize optimizer with only the trainable set of parameters
optimizer = optimizer.init(model.trainable_parameters())
...

@jax.jit
def train_step(model, x, y, optimizer):
    # only differentiate w.r.t. parameters whose module is not frozen
    params = model.trainable_parameters()
    (loss, model), grads = loss_fn(params, model, x, y)

    ...
```
