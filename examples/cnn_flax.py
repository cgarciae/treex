import typing as tp
from functools import partial

import dataget
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from flax import linen
from tqdm import tqdm

import treex as tx

Batch = tp.Mapping[str, np.ndarray]
np.random.seed(420)


def loss_fn(
    params: tx.FlaxModule, model: tx.FlaxModule, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[jnp.ndarray, tp.Tuple[tx.FlaxModule, jnp.ndarray]]:
    model = model.update(params)
    y_pred = model(x, training=model.training)

    loss = jnp.mean(
        optax.softmax_cross_entropy(
            y_pred,
            jax.nn.one_hot(y, 10),
        )
    )

    acc_batch = y_pred.argmax(axis=1) == y

    return loss, (model, acc_batch)


@jax.jit
def train_step(
    model: tx.FlaxModule, optimizer: tx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[jnp.ndarray, tx.FlaxModule, tx.Optimizer, jnp.ndarray]:
    params = model.filter(tx.Parameter)

    (loss, (model, acc_batch)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, model, x, y
    )

    params = optimizer.apply_updates(grads, params)
    model = model.update(params)

    return loss, model, optimizer, acc_batch


@jax.jit
def test_step(
    model: tx.FlaxModule, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:

    loss, (model, acc_batch) = loss_fn(model, model, x, y)

    return loss, acc_batch


@jax.jit
def predict(model: tx.FlaxModule, x: jnp.ndarray):
    return model(x, model.training).argmax(axis=1)


class CNN(linen.Module):
    @linen.compact
    def __call__(self, x, training):
        x = linen.Conv(32, [3, 3], strides=[2, 2])(x)
        x = linen.BatchNorm(use_running_average=not training)(x)
        x = linen.Dropout(0.05, deterministic=not training)(x)
        x = jax.nn.relu(x)
        x = linen.Conv(64, [3, 3], strides=[2, 2])(x)
        x = linen.BatchNorm(use_running_average=not training)(x)
        x = linen.Dropout(0.1, deterministic=not training)(x)
        x = jax.nn.relu(x)
        x = linen.Conv(128, [3, 3], strides=[2, 2])(x)
        x = x.mean(axis=[1, 2])  # GlobalAveragePooling2D
        x = linen.Dense(10)(x)

        return x


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 32,
    steps_per_epoch: int = -1,
):
    # load data
    X_train, y_train, X_test, y_test = dataget.image.mnist().get()
    X_train = X_train[..., None]
    X_test = X_test[..., None]

    model = tx.FlaxModule(
        CNN(),
        sample_inputs=tx.Inputs(X_train[:32], training=True),
    ).init(42)

    print(model.tabulate())

    optimizer = tx.Optimizer(optax.adamw(1e-3))
    optimizer = optimizer.init(model.filter(tx.Parameter))

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
                len(X_train) // batch_size if steps_per_epoch < 1 else steps_per_epoch
            ),
            desc="training",
            unit="batch",
            leave=False,
        ):
            idx = np.random.choice(len(X_train), batch_size)
            x = X_train[idx]
            y = y_train[idx]
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
                len(X_test) // batch_size if steps_per_epoch < 1 else steps_per_epoch
            ),
            desc="testing",
            unit="batch",
            leave=False,
        ):
            idx = np.random.choice(len(X_test), batch_size)
            x = X_test[idx]
            y = y_test[idx]
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

    y_pred = predict(model, x_sample)

    plt.figure()
    for i in range(5):
        ax: plt.Axes = plt.subplot(2, 5, i + 1)
        ax.set_title(f"{y_pred[i]}")
        plt.imshow(x_sample[i], cmap="gray")
        ax: plt.Axes = plt.subplot(2, 5, 5 + i + 1)
        ax.set_title(f"{y_pred[5 + i]}")
        plt.imshow(x_sample[5 + i], cmap="gray")

    plt.show()
    plt.close()


if __name__ == "__main__":

    typer.run(main)
