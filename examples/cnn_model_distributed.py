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
from numpy.core.fromnumeric import reshape
from tqdm import tqdm

import treex as tx

Batch = tp.Mapping[str, np.ndarray]
Module = tx.Sequential
LossesAndMetrics = tx.metrics.LossesAndMetrics
np.random.seed(420)

M = tp.TypeVar("M", bound="Model")
A = tp.TypeVar("A")


class Model(tx.Module):
    key: jnp.ndarray = tx.node()
    module: Module
    optimizer: tx.Optimizer
    losses_and_metrics: LossesAndMetrics

    def __init__(
        self,
        key: tp.Union[jnp.ndarray, int],
        module: Module,
        optimizer: optax.GradientTransformation,
        losses_and_metrics: LossesAndMetrics,
    ) -> None:
        self.key = tx.Key(key)
        self.module = module
        self.optimizer = tx.Optimizer(optimizer)
        self.losses_and_metrics = losses_and_metrics

    def __call__(self, *args, **kwargs) -> tp.Any:
        return self.module(*args, **kwargs)

    @partial(
        jax.pmap,
        in_axes=(None, None, 0),
        out_axes=0,
        axis_name="device",
    )
    @tx.toplevel_mutable
    def init_step(
        self: M,
        x: tp.Any,
        device_idx: jnp.ndarray,
    ) -> M:

        init_key, self.key = jax.random.split(self.key)
        self.module = self.module.init(init_key, x)
        self.optimizer = self.optimizer.init(self.module.parameters())
        self.losses_and_metrics = self.losses_and_metrics.reset()

        # assign unique rng keys
        self.key = jax.random.fold_in(self.key, jax.lax.axis_index("device"))

        return self

    @partial(jax.pmap, axis_name="device")
    def reset_step(self: M) -> M:
        return self.replace(
            losses_and_metrics=self.losses_and_metrics.reset(),
        )

    @tx.toplevel_mutable
    def loss_fn(
        self: "Model",
        params: tp.Optional[Module],
        key: tp.Optional[jnp.ndarray],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tp.Tuple[jnp.ndarray, "Model"]:

        if params is not None:
            self.module = self.module.merge(params)

        preds, self.module = self.module.apply(key, x)

        batch_updates: LossesAndMetrics = self.losses_and_metrics.batch_updates(
            target=y,
            preds=preds,
        )
        loss = batch_updates.total_loss()

        # sync updates between devices
        batch_updates = jax.lax.all_gather(batch_updates, axis_name="device")
        batch_updates = batch_updates.aggregate()
        self.losses_and_metrics = self.losses_and_metrics.merge(batch_updates)

        return loss, self

    @partial(jax.pmap, axis_name="device")
    @tx.toplevel_mutable
    def train_step(
        self: M,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> M:
        print("JITTTTING")
        params = self.module.parameters()
        loss_key, self.key = jax.random.split(self.key)

        grads, self = jax.grad(self.loss_fn, has_aux=True)(params, loss_key, x, y)

        grads = jax.lax.pmean(grads, axis_name="device")

        params, self.optimizer = self.optimizer.update(grads, params)
        self.module = self.module.merge(params)

        # sync batch statistics
        pmean = partial(jax.lax.pmean, axis_name="device")
        self.module = self.module.map(pmean, tx.BatchStat)

        return self

    @partial(jax.pmap, axis_name="device")
    @tx.toplevel_mutable
    def test_step(
        self: M,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> M:
        loss, self = self.loss_fn(None, None, x, y)

        return self

    @jax.jit
    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.module(x).argmax(axis=1)


def to_local(x: A) -> A:
    return x[0]


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 32,
    steps_per_epoch: int = -1,
    use_tpu: bool = False,
    seed: int = 42,
    lr: float = 1e-3,
):

    if use_tpu:
        print("Setting up TPU...")
        jax.tools.colab_tpu.setup_tpu()

    n_devices = jax.device_count()
    device_idx = jnp.arange(n_devices)

    # load data
    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = np.stack(dataset["train"]["image"])[..., None]
    y_train = dataset["train"]["label"]
    X_test = np.stack(dataset["test"]["image"])[..., None]
    y_test = dataset["test"]["label"]

    # define model
    model = Model(
        key=seed,
        module=tx.Sequential(
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
        ),
        optimizer=optax.adamw(lr),
        losses_and_metrics=tx.LossesAndMetrics(
            metrics=tx.metrics.Accuracy(),
            losses=tx.losses.Crossentropy(),
        ),
    )

    model: Model = model.init_step(X_train[:batch_size], device_idx)

    print(
        model.module.map(to_local).tabulate(X_train[:batch_size], show_signatures=True)
    )

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
        model = model.reset_step()
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
            model = model.train_step(x, y)

        train_logs = model.losses_and_metrics.map(to_local).compute()
        history_train.append(train_logs)

        # ---------------------------------------
        # test
        # ---------------------------------------
        model = model.eval()
        model = model.reset_step()
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

            model = model.test_step(x, y)

        test_logs = model.losses_and_metrics.map(to_local).compute()
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

    preds = model.predict(x_sample)

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
