import typing as tp
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_metrics as jm
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets.load import load_dataset
from tqdm import tqdm

import treex as tx


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x, training):
        x = nn.Conv(32, [3, 3], strides=[2, 2])(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.Dropout(0.05, deterministic=not training)(x)
        x = jax.nn.relu(x)
        x = nn.Conv(64, [3, 3], strides=[2, 2])(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.Dropout(0.1, deterministic=not training)(x)
        x = jax.nn.relu(x)
        x = nn.Conv(128, [3, 3], strides=[2, 2])(x)
        x = x.mean(axis=(1, 2))  # GlobalAveragePooling2D
        x = nn.Dense(10)(x)

        return x


Batch = tp.Mapping[str, np.ndarray]
Model = tx.FlaxModule[CNN]
Logs = tp.Dict[str, jnp.ndarray]
np.random.seed(420)


@partial(jax.jit, static_argnums=(2,))
def init_step(
    model: Model,
    optiizer: tx.Optimizer,
    seed: int,
    inputs: tp.Any,
) -> tp.Tuple[Model, tx.Optimizer]:
    model = model.init(seed, inputs)
    optiizer = optiizer.init(model.parameters())

    return model, optiizer


@jax.jit
def reset_step(losses_and_metrics: jm.LossesAndMetrics) -> jm.LossesAndMetrics:
    return losses_and_metrics.reset()


def loss_fn(
    params: tp.Optional[Model],
    key: tp.Optional[jnp.ndarray],
    model: Model,
    losses_and_metrics: jm.LossesAndMetrics,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, tp.Tuple[Model, jm.LossesAndMetrics]]:
    if params is not None:
        model = model.merge(params)

    preds, model = model.apply(key, x)
    loss, losses_and_metrics = losses_and_metrics.loss_and_update(target=y, preds=preds)

    return loss, (model, losses_and_metrics)


@jax.jit
def train_step(
    key: jnp.ndarray,
    model: Model,
    optimizer: tx.Optimizer,
    losses_and_metrics: jm.LossesAndMetrics,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tp.Tuple[Model, tx.Optimizer, jm.LossesAndMetrics]:
    print("JITTTTING")
    params = model.parameters()

    grads, (model, losses_and_metrics) = jax.grad(loss_fn, has_aux=True)(
        params, key, model, losses_and_metrics, x, y
    )

    params, optimizer = optimizer.update(grads, params)
    model = model.merge(params)

    return model, optimizer, losses_and_metrics


@jax.jit
def test_step(
    model: Model,
    losses_and_metrics: jm.LossesAndMetrics,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> jm.LossesAndMetrics:
    key = tx.Key(42)
    loss, (model, losses_and_metrics) = loss_fn(
        None, key, model, losses_and_metrics, x, y
    )

    return losses_and_metrics


@jax.jit
def predict(model: Model, x: jnp.ndarray):
    key = tx.Key(42)
    return model.apply(key, x)[0].argmax(axis=1)


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 32,
    steps_per_epoch: int = -1,
):

    # load data
    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = np.stack(dataset["train"]["image"])[..., None]
    y_train = dataset["train"]["label"]
    X_test = np.stack(dataset["test"]["image"])[..., None]
    y_test = dataset["test"]["label"]

    # define model
    model: Model = tx.FlaxModule(CNN()).init(42, X_train[:32])

    optimizer = tx.Optimizer(optax.adamw(1e-3))
    losses_and_metrics: jm.LossesAndMetrics = jm.LossesAndMetrics(
        losses=jm.losses.Crossentropy(),
        metrics=jm.metrics.Accuracy(),
    )

    model, optimizer = init_step(model, optimizer, seed=42, inputs=X_train[:batch_size])

    print(model.tabulate(X_train[:batch_size], show_signatures=True))

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    train_logs = {}
    test_logs = {}

    history_train: tp.List[Logs] = []
    history_test: tp.List[Logs] = []
    key = tx.Key(42)

    for epoch in range(epochs):
        # ---------------------------------------
        # train
        # ---------------------------------------
        model = model.train()
        losses_and_metrics = reset_step(losses_and_metrics)
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
            key, step_key = jax.random.split(key)
            model, optimizer, losses_and_metrics = train_step(
                step_key, model, optimizer, losses_and_metrics, x, y
            )

        train_logs = losses_and_metrics.compute()
        history_train.append(train_logs)

        # ---------------------------------------
        # test
        # ---------------------------------------
        model = model.eval()
        losses_and_metrics = reset_step(losses_and_metrics)
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
            losses_and_metrics = test_step(model, losses_and_metrics, x, y)

        test_logs = losses_and_metrics.compute()
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
