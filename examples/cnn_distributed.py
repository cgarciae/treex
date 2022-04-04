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
from treex import metrics

Model = tx.Sequential
Batch = tp.Mapping[str, np.ndarray]
A = tp.TypeVar("A")
np.random.seed(420)


@partial(
    jax.pmap,
    in_axes=(None, None, None, None, None, 0),
    axis_name="device",
)
def init_step(
    key: jnp.ndarray,
    model: Model,
    optimizer: tx.Optimizer,
    losses_and_metrics: tx.LossesAndMetrics,
    x: tp.Any,
    device_idx: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, Model, tx.Optimizer, tx.LossesAndMetrics]:

    model_key, key = jax.random.split(key)
    model = model.init(model_key, x)
    optimizer = optimizer.init(model.trainable_parameters())
    losses_and_metrics = losses_and_metrics.reset()

    # assign unique rng keys
    axis_index = jax.lax.axis_index("device")
    key = jax.random.fold_in(key, axis_index)

    return key, model, optimizer, losses_and_metrics


@partial(jax.pmap, axis_name="device")
def reset_step(losses_and_metrics: tx.LossesAndMetrics) -> tx.LossesAndMetrics:
    return losses_and_metrics.reset()


def loss_fn(
    params: tp.Optional[Model],
    key: tp.Optional[jnp.ndarray],
    model: Model,
    losses_and_metrics: metrics.LossesAndMetrics,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, tp.Tuple[Model, tx.LossesAndMetrics]]:

    if params is not None:
        model = model.merge(params)

    preds, model = model.apply(key, x)

    batch_updates = losses_and_metrics.batch_updates(target=y, preds=preds)
    loss = batch_updates.total_loss()

    # sync updates between devices
    losses_and_metrics = (
        jax.lax.all_gather(batch_updates, axis_name="device")
        .aggregate()
        .merge(losses_and_metrics)
    )

    return loss, (model, losses_and_metrics)


@partial(jax.pmap, axis_name="device")
def train_step(
    key: jnp.ndarray,
    model: Model,
    optimizer: tx.Optimizer,
    losses_and_metrics: tx.LossesAndMetrics,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, Model, tx.Optimizer, tx.LossesAndMetrics]:
    params = model.trainable_parameters()
    loss_key, key = jax.random.split(key)

    grads, (model, losses_and_metrics) = jax.grad(loss_fn, has_aux=True)(
        params, loss_key, model, losses_and_metrics, x, y
    )

    # sync gradients across devices
    grads = jax.lax.pmean(grads, axis_name="device")

    params, optimizer = optimizer.update(grads, params)
    model = model.merge(params)

    # sync batch statistics
    pmean = partial(jax.lax.pmean, axis_name="device")
    model = model.map(pmean, tx.BatchStat)

    key = tp.cast(jnp.ndarray, key)

    return key, model, optimizer, losses_and_metrics


@partial(jax.pmap, axis_name="device")
def test_step(
    model: Model,
    losses_and_metrics: tx.LossesAndMetrics,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tx.LossesAndMetrics:

    _, (_, losses_and_metrics) = loss_fn(None, None, model, losses_and_metrics, x, y)

    return losses_and_metrics


@jax.jit
def predict(model: Model, x: jnp.ndarray):
    return model(x).argmax(axis=1)


def to_local(x: A) -> A:
    return x[0]


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

    # load data
    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = np.stack(dataset["train"]["image"])
    y_train = dataset["train"]["label"]
    X_test = np.stack(dataset["test"]["image"])
    y_test = dataset["test"]["label"]

    X_train = X_train[..., None]
    X_test = X_test[..., None]

    # define model
    model: tx.Sequential = tx.Sequential(
        tx.Conv(32, [3, 3], strides=[2, 2]),
        tx.BatchNorm(),
        tx.Dropout(0.05),
        jax.nn.relu,
        tx.Conv(64, [3, 3], strides=[2, 2]),
        tx.BatchNorm(),
        tx.Dropout(0.1),
        jax.nn.relu,
        tx.Conv(128, [3, 3], strides=[2, 2]),
        partial(jnp.mean, axis=(1, 2)),
        tx.Linear(10),
    )
    optimizer = tx.Optimizer(optax.adamw(1e-3))
    losses_and_metrics: tx.LossesAndMetrics = tx.LossesAndMetrics(
        losses=tx.losses.Crossentropy(),
        metrics=tx.metrics.Accuracy(),
    )
    key = tx.Key(42)

    key, model, optimizer, losses_and_metrics = init_step(
        key, model, optimizer, losses_and_metrics, X_train[:batch_size], device_idx
    )

    print(model.map(to_local).tabulate(X_train[:batch_size], show_signatures=True))

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    train_logs = {}
    test_logs = {}
    history_train = []
    history_test = []

    for epoch in range(epochs):
        # ---------------------------------------
        # train
        # ---------------------------------------
        model = model.train()
        losses_and_metrics = reset_step(losses_and_metrics)
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

            key, model, optimizer, losses_and_metrics = train_step(
                key, model, optimizer, losses_and_metrics, x, y
            )

        train_logs = losses_and_metrics.map(to_local).compute()
        history_train.append(train_logs)

        # ---------------------------------------
        # test
        # ---------------------------------------
        model = model.eval()
        losses_and_metrics = reset_step(losses_and_metrics)
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

            losses_and_metrics = test_step(model, losses_and_metrics, x, y)

        test_logs = losses_and_metrics.map(to_local).compute()
        history_test.append(test_logs)

        test_logs = {f"{name}_valid": value for name, value in test_logs.items()}
        logs = {**train_logs, **test_logs}
        logs = {name: float(value) for name, value in logs.items()}

        print(f"[{epoch}] {logs}")

    # pass params from the first device to host
    model = model.map(to_local).eval()

    # plot logs
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
