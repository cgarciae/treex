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

Batch = tp.Mapping[str, np.ndarray]
Model = tx.Sequential
Logs = tp.Dict[str, jnp.ndarray]
np.random.seed(420)


@partial(jax.jit, static_argnums=(2,))
def init_step(
    model: Model, optiizer: tx.Optimizer, seed: int
) -> tp.Tuple[Model, tx.Optimizer]:
    model = model.init(seed)
    optiizer = optiizer.init(model.parameters())

    return model, optiizer


@jax.jit
def reset_step(loss_logs: tx.LossAndLogs) -> tx.LossAndLogs:
    loss_logs.reset()
    return loss_logs


def loss_fn(
    params: Model,
    model: Model,
    loss_logs: tx.LossAndLogs,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, tp.Tuple[Model, tx.LossAndLogs, Logs]]:
    model = model.merge(params)
    y_pred = model(x)
    loss, losses_logs, metrics_logs = loss_logs.batch_loss_epoch_logs(
        y_true=y, y_pred=y_pred
    )
    logs = {**losses_logs, **metrics_logs}

    return loss, (model, loss_logs, logs)


@jax.jit
def train_step(
    model: Model,
    optimizer: tx.Optimizer,
    loss_logs: tx.LossAndLogs,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tp.Tuple[Logs, Model, tx.Optimizer, tx.LossAndLogs]:
    print("JITTTTING")
    params = model.parameters()

    grads, (model, loss_logs, logs) = jax.grad(loss_fn, has_aux=True)(
        params, model, loss_logs, x, y
    )

    params = optimizer.update(grads, params)
    model = model.merge(params)

    return logs, model, optimizer, loss_logs


@jax.jit
def test_step(
    model: Model, loss_logs: tx.LossAndLogs, x: jnp.ndarray, y: jnp.ndarray
) -> tp.Tuple[Logs, tx.LossAndLogs]:

    loss, (model, loss_logs, logs) = loss_fn(model, model, loss_logs, x, y)

    return logs, loss_logs


@jax.jit
def predict(model: Model, x: jnp.ndarray):
    return model(x).argmax(axis=1)


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 32,
    steps_per_epoch: int = -1,
):

    model: Model = tx.Sequential(
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
    loss_logs = tx.LossAndLogs(
        losses=tx.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tx.metrics.Accuracy(),
    )

    model, optimizer = init_step(model, optimizer, seed=42)

    # load data
    X_train, y_train, X_test, y_test = dataget.image.mnist().get()
    X_train = X_train[..., None]
    X_test = X_test[..., None]

    print(model.tabulate(X_train[:batch_size], signature=True))

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    train_logs = {}
    test_logs = {}

    history_train: tp.List[Logs] = []
    history_test: tp.List[Logs] = []

    for epoch in range(epochs):
        # ---------------------------------------
        # train
        # ---------------------------------------
        model = model.train()
        loss_logs = reset_step(loss_logs)
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
            train_logs, model, optimizer, loss_logs = train_step(
                model, optimizer, loss_logs, x, y
            )

        history_train.append(train_logs)

        # ---------------------------------------
        # test
        # ---------------------------------------
        model = model.eval()
        loss_logs = reset_step(loss_logs)
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
            test_logs, loss_logs = test_step(model, loss_logs, x, y)

        history_test.append(test_logs)
        test_logs = {f"{name}_valid": value for name, value in test_logs.items()}

        logs = {**train_logs, **test_logs}
        logs = {name: float(value) for name, value in logs.items()}

        print(f"[{epoch}] {logs}")

    model = model.eval()

    for name in history_train[0]:
        plt.figure()
        plt.title(name)
        plt.plot([logs[name] for logs in history_train])
        plt.plot([logs[name] for logs in history_test])

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
