import typing as tp
from functools import partial

import einops
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets.load import load_dataset
from tqdm import tqdm

import treex as tx

Model = tx.Sequential
Batch = tp.Mapping[str, np.ndarray]
np.random.seed(420)


def loss_fn(
    params: Model, model: Model, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[jnp.ndarray, tp.Tuple[Model, jnp.ndarray]]:
    model = model.merge(params)
    preds = model(x)

    loss = jnp.mean(
        optax.softmax_cross_entropy(
            preds,
            jax.nn.one_hot(y, 10),
        )
    )

    acc_batch = preds.argmax(axis=1) == y

    acc_batch = jax.lax.all_gather(acc_batch, axis_name="device")

    return loss, (model, acc_batch)


@partial(jax.pmap, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, None), axis_name="device")
def train_step(
    model: Model, optimizer: tx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[jnp.ndarray, Model, tx.Optimizer, jnp.ndarray]:
    params = model.filter(tx.Parameter)

    (loss, (model, acc_batch)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, model, x, y
    )

    # sync gradients across devices
    grads = jax.lax.pmean(grads, axis_name="device")

    params = optimizer.update(grads, params)
    model = model.merge(params)

    # sync batch statistics
    model = model.map(partial(jax.lax.pmean, axis_name="device"), tx.BatchStat)

    return loss, model, optimizer, acc_batch


@partial(jax.pmap, in_axes=(0, 0, 0), out_axes=(0, None), axis_name="device")
def test_step(
    model: Model, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:

    loss, (model, acc_batch) = loss_fn(model, model, x, y)

    return loss, acc_batch


@partial(
    jax.pmap, in_axes=(None, None, None, None, 0), out_axes=(0, 0), axis_name="device"
)
def init_step(
    model: Model,
    optimizer: tx.Optimizer,
    key: jnp.ndarray,
    inputs: tp.Any,
    device_idx: jnp.ndarray,
) -> tp.Tuple[Model, tx.Optimizer]:

    model = model.init(key, inputs=inputs)
    optimizer = optimizer.init(model.filter(tx.Parameter))

    # assign unique rng keys
    axis_index = jax.lax.axis_index("device")
    model = model.map(lambda k: jax.random.fold_in(k, axis_index), tx.Rng)

    return model, optimizer


@jax.jit
def predict(model: Model, x: jnp.ndarray):
    return model(x).argmax(axis=1)


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 32,
    steps_per_epoch: int = -1,
    use_tpu: bool = False,
):

    if use_tpu:
        print("Setting up TPU...")
        jax.tools.colab_tpu.setup_tpu()

    n_devices = jax.device_count()
    device_idx = jnp.arange(n_devices)
    key = tx.Key(42)

    # load data
    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = dataset["train"]["image"]
    y_train = dataset["train"]["label"]
    X_test = dataset["test"]["image"]
    y_test = dataset["test"]["label"]

    X_train = X_train[..., None]
    X_test = X_test[..., None]

    # define model
    model = tx.Sequential(
        tx.Conv(32, [3, 3], strides=[2, 2]),
        tx.BatchNorm(),
        tx.Dropout(0.05),
        jax.nn.relu,
        tx.Conv(64, [3, 3], strides=[2, 2]),
        tx.BatchNorm(),
        tx.Dropout(0.1),
        jax.nn.relu,
        tx.Conv(128, [3, 3], strides=[2, 2]),
        partial(jnp.mean, axis=[1, 2]),
        tx.Linear(10),
    )
    optimizer = tx.Optimizer(optax.adamw(1e-3))

    model, optimizer = init_step(
        model, optimizer, key, X_train[:batch_size], device_idx
    )

    print(model.tabulate(signature=True))

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    epoch_train_losses = []
    epoch_train_accs = []
    epoch_test_losses = []
    epoch_test_accs = []

    for epoch in range(epochs):
        # ---------------------------------------
        # train
        # ---------------------------------------
        train_losses = []
        train_accs = []
        model = model.train()
        for step in tqdm(
            range(
                len(X_train) // (batch_size * n_devices)
                if steps_per_epoch < 1
                else steps_per_epoch
            ),
            desc="training",
            unit="batch",
            leave=False,
        ):
            idx = np.random.choice(len(X_train), batch_size * n_devices)
            x = einops.rearrange(
                X_train[idx], "(device batch) ... -> device batch ...", device=n_devices
            )
            y = einops.rearrange(
                y_train[idx], "(device batch) ... -> device batch ...", device=n_devices
            )

            loss, model, optimizer, acc = train_step(model, optimizer, x, y)
            train_losses.append(loss)
            train_accs.append(acc)

        epoch_train_loss = jnp.mean(jnp.stack(train_losses))
        epoch_train_acc = jnp.mean(jnp.stack(train_accs))
        epoch_train_losses.append(epoch_train_loss)
        epoch_train_accs.append(epoch_train_acc)

        # ---------------------------------------
        # test
        # ---------------------------------------
        test_losses = []
        test_accs = []
        model = model.eval()
        for step in tqdm(
            range(
                len(X_test) // (batch_size * n_devices)
                if steps_per_epoch < 1
                else steps_per_epoch
            ),
            desc="testing",
            unit="batch",
            leave=False,
        ):
            idx = np.random.choice(len(X_test), batch_size * n_devices)
            x = einops.rearrange(
                X_test[idx], "(device batch) ... -> device batch ...", device=n_devices
            )
            y = einops.rearrange(
                y_test[idx], "(device batch) ... -> device batch ...", device=n_devices
            )

            loss, acc = test_step(model, x, y)
            test_losses.append(loss)
            test_accs.append(acc)

        epoch_test_loss = jnp.mean(jnp.stack(test_losses))
        epoch_test_acc = jnp.mean(jnp.stack(test_accs))
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_accs.append(epoch_test_acc)

        print(
            f"[{epoch}] loss_train={epoch_train_loss}, acc_train={epoch_train_acc}, loss_test={epoch_test_loss}, acc_test={epoch_test_acc}"
        )

    # pass params from the first device to host
    model = model.map(lambda x: x[0])

    model = model.eval()

    # plot loss curve
    plt.figure()
    plt.plot(epoch_train_losses)
    plt.plot(epoch_test_losses)

    # plot acc curve
    plt.figure()
    plt.plot(epoch_train_accs)
    plt.plot(epoch_test_accs)

    # visualize reconstructions
    idxs = np.random.choice(len(X_test), 10)
    x_sample = X_test[idxs]

    preds = predict(model, x_sample)

    plt.figure()
    for i in range(5):
        ax: plt.Axes = plt.subplot(2, 5, i + 1)
        ax.set_title(f"{preds[i]}")
        plt.imshow(x_sample[i], cmap="gray")
        ax: plt.Axes = plt.subplot(2, 5, 5 + i + 1)
        ax.set_title(f"{preds[5 + i]}")
        plt.imshow(x_sample[5 + i], cmap="gray")

    plt.show()
    plt.close()


if __name__ == "__main__":
    typer.run(main)
