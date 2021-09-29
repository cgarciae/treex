import typing as tp
from functools import partial

import dataget
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from tqdm import tqdm

import treex as tx
from treex import metrics
from treex.utils import _check_rejit

Batch = tp.Mapping[str, np.ndarray]
Model = tx.Sequential
Metric = tx.metrics.Accuracy
np.random.seed(420)


@partial(jax.jit, static_argnums=(2,))
def init_step(
    model: Model, optiizer: tx.Optimizer, seed: int
) -> tp.Tuple[Model, tx.Optimizer]:
    model = model.init(seed)
    optiizer = optiizer.init(model.parameters())

    return model, optiizer


@jax.jit
def reset_step(metric: Metric) -> Metric:
    metric.reset()
    return metric


def loss_fn(
    params: Model,
    model: Model,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, tp.Tuple[Model, jnp.ndarray]]:
    model = model.merge(params)
    y_pred = model(x)

    loss = jnp.mean(
        optax.softmax_cross_entropy(
            y_pred,
            jax.nn.one_hot(y, 10),
        )
    )

    return loss, (model, y_pred)


@jax.jit
def train_step(
    model: Model,
    optimizer: tx.Optimizer,
    metric: Metric,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, Model, tx.Optimizer, Metric]:
    print("JITTTTING")
    params = model.parameters()

    (loss, (model, y_pred)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, model, x, y
    )

    params = optimizer.update(grads, params)
    model = model.merge(params)
    _batch_metric = metric(y_true=y, y_pred=y_pred)

    return loss, model, optimizer, metric


@jax.jit
def test_step(
    model: Model, metric: Metric, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[jnp.ndarray, Metric]:

    loss, (model, y_pred) = loss_fn(model, model, x, y)

    _batch_metric = metric(y_true=y, y_pred=y_pred)

    return loss, metric


@jax.jit
def predict(model: Model, x: jnp.ndarray):
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
        partial(jnp.mean, axis=[1, 2]),
        tx.Linear(128, 10),
    )

    optimizer = tx.Optimizer(optax.adamw(1e-3))
    metric = tx.metrics.Accuracy(argmax_preds=True)

    model, optimizer = init_step(model, optimizer, seed=42)

    # load data
    X_train, y_train, X_test, y_test = dataget.image.mnist().get()
    X_train = X_train[..., None]
    X_test = X_test[..., None]

    print(model.tabulate(X_train[:batch_size], signature=True))
    print(tx.to_string(model, static_fields=False))

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
        model = model.train()
        metric = reset_step(metric)
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
            loss, model, optimizer, metric = train_step(model, optimizer, metric, x, y)
            train_losses.append(loss)

        epoch_train_loss = jnp.mean(jnp.stack(train_losses))
        epoch_train_losses.append(epoch_train_loss)
        epoch_train_accs.append(metric.compute())

        # ---------------------------------------
        # test
        # ---------------------------------------
        test_losses = []
        model = model.eval()
        metric = reset_step(metric)
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
            loss, metric = test_step(model, metric, x, y)
            test_losses.append(loss)

        epoch_test_loss = jnp.mean(jnp.stack(test_losses))
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_accs.append(metric.compute())

        print(
            f"[{epoch}] loss_train={epoch_train_loss}, acc_train={epoch_train_accs[-1]}, loss_test={epoch_test_loss}, acc_test={epoch_test_accs[-1]}"
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
