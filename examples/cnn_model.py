import functools
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets.load import load_dataset
from tqdm import tqdm

import treex as tx
from treex import metrics
from treex.utils import _check_rejit

Batch = tp.Mapping[str, np.ndarray]
Module = tx.Sequential
Metric = tx.metrics.Accuracy
Logs = tp.Mapping[str, jnp.ndarray]
np.random.seed(420)

M = tp.TypeVar("M", bound="Model")
C = tp.TypeVar("C", bound="tp.Callable")


class Model(tx.Module):
    key: jnp.ndarray = tx.node()

    def __init__(
        self,
        key: tp.Union[jnp.ndarray, int],
        module: Module,
        optimizer: optax.GradientTransformation,
        losses: tp.Any,
        metrics: tp.Any,
    ) -> None:
        self.key = tx.Key(key)
        self.module = module
        self.optimizer = tx.Optimizer(optimizer)
        self.losses_and_metrics = tx.LossesAndMetrics(
            losses=losses,
            metrics=metrics,
        )

    def __call__(self, *args, **kwargs) -> tp.Any:
        return self.module(*args, **kwargs)

    @jax.jit
    @tx.toplevel_mutable
    def init_step(self: M, x: tp.Any) -> M:

        init_key, self.key = jax.random.split(self.key)
        self.module = self.module.init(init_key, x)
        self.optimizer = self.optimizer.init(self.module.parameters())
        self.losses_and_metrics = self.losses_and_metrics.reset()

        return self

    @jax.jit
    @tx.toplevel_mutable
    def reset_step(self: M) -> M:
        self.losses_and_metrics = self.losses_and_metrics.reset()
        return self

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

        loss, self.losses_and_metrics = self.losses_and_metrics.loss_and_update(
            target=y,
            preds=preds,
        )

        return loss, self

    @jax.jit
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

        params, self.optimizer = self.optimizer.update(grads, params)
        self.module = self.module.merge(params)

        return self

    @jax.jit
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


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 32,
    steps_per_epoch: int = -1,
    seed: int = 420,
):

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
        optimizer=optax.adamw(1e-3),
        losses=tx.losses.Crossentropy(),
        metrics=tx.metrics.Accuracy(),
    )

    model: Model = model.init_step(X_train[:batch_size])

    print(model.module.tabulate(X_train[:batch_size], show_signatures=True))

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
        model = model.reset_step()
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
            model = model.train_step(x, y)

            # jax.tree_map(lambda a, b: a, model, model2)

            train_logs = model.losses_and_metrics.compute()

        history_train.append(train_logs)

        # ---------------------------------------
        # test
        # ---------------------------------------
        model = model.eval()
        model = model.reset_step()
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
            model = model.test_step(x, y)

        test_logs = model.losses_and_metrics.compute()
        history_test.append(test_logs)

        test_logs = {f"{name}_valid": value for name, value in test_logs.items()}
        logs = {**train_logs, **test_logs}
        logs = {name: float(value) for name, value in logs.items()}

        print(f"[{epoch}] {logs}")

    model = model.eval()

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
