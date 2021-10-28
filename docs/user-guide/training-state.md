
# Training State

Modules have a `training: bool` property that specifies whether the module is in training mode or not. This property conditions the behavior of Modules such as `Dropout` and `BatchNorm`, which behave differently between training and evaluation. 

```python hl_lines="6"
# training loop
for step in range(1000):
    loss, model, opt_state = train_step(model, x, y, opt_state)

# prepare for evaluation
model = model.eval()

# make predictions
preds = model(X_test)
```

To switch between modes, use the `.train()` and `.eval()` methods, they return a new Module whose `training` state and the state of all of its submodules (recursively) are set to the desired value.