import typing as tp
from functools import partial

import dataget
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer

import treex as tx

Batch = tp.Mapping[str, np.ndarray]
np.random.seed(420)


@partial(jax.value_and_grad, has_aux=True)
def loss_fn(
    params: tx.Sequential, model: tx.Sequential, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[jnp.ndarray, tp.Tuple[tx.Sequential, jnp.ndarray]]:
    model = model.update(params)
    y_pred = model(x)

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
    model: tx.Sequential, optimizer: tx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[jnp.ndarray, tx.Sequential, tx.Optimizer, jnp.ndarray]:
    params = model.filter(tx.Parameter)
    (loss, (model, acc_batch)), grads = loss_fn(params, model, x, y)

    params = optimizer.apply_updates(grads, params)
    model = model.update(params)

    return loss, model, optimizer, acc_batch


@jax.jit
def predict(model: tx.Sequential, x: jnp.ndarray):
    print("JITTING")
    return model(x).argmax(axis=1)


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 32,
    steps_per_epoch: int = -1,
):

    model = tx.Sequential(
        tx.Conv(1, 32, [3, 3], strides=[2, 2]),
        tx.BatchNorm(32),
        tx.Dropout(0.05),
        jax.nn.relu,
        tx.Conv(32, 64, [3, 3], strides=[2, 2]),
        tx.BatchNorm(64),
        tx.Dropout(0.1),
        jax.nn.relu,
        tx.Conv(64, 128, [3, 3], strides=[2, 2]),
        lambda x: x.mean(axis=[1, 2]),  # GlobalAveragePooling2D
        tx.Linear(128, 10),
    ).init(42)

    print(model.tabulate())

    optimizer = tx.Optimizer(optax.adamw(1e-3))
    optimizer = optimizer.init(model.filter(tx.Parameter))

    # load data
    X_train, y_train, X_test, y_test = dataget.image.mnist().get()
    X_train = X_train[..., None]
    X_test = X_test[..., None]

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    epoch_losses = []
    epoch_accs = []

    for epoch in range(epochs):
        losses = []
        accs = []
        model = model.train()
        for step in range(
            len(X_train) // batch_size if steps_per_epoch < 1 else steps_per_epoch
        ):
            idx = np.random.choice(len(X_train), batch_size)
            x = X_train[idx]
            y = y_train[idx]
            loss, model, optimizer, acc_batch = train_step(model, optimizer, x, y)
            losses.append(loss)
            accs.append(acc_batch)

        epoch_loss = jnp.mean(jnp.stack(losses))
        epoch_acc = jnp.mean(jnp.stack(accs))
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)
        print(f"[{epoch}] loss={epoch_loss}, acc={epoch_acc}")

    model = model.eval()

    # plot loss curve
    plt.figure()
    plt.plot(epoch_losses)

    # plot acc curve
    plt.figure()
    plt.plot(epoch_accs)

    # visualize reconstructions
    idxs = np.random.choice(10, len(X_test))
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
