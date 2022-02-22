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


class Model(tx.Module):
    def __init__(
        self,
        module: Module,
        optimizer: optax.GradientTransformation,
        losses: tp.Any,
        metrics: tp.Any,
    ) -> None:
        self.module = module
        self.optimizer = tx.Optimizer(optimizer)
        self.loss_and_logs = tx.LossAndLogs(
            losses=losses,
            metrics=metrics,
        )

    def __call__(self, *args, **kwargs) -> tp.Any:
        return self.module(*args, **kwargs)

    @partial(jax.jit, static_argnums=(1,))
    def init_step(self: M, seed: int, inputs: tp.Any) -> M:
        self.module = self.module.init(seed, inputs=inputs)
        self.optimizer = self.optimizer.init(self.module.parameters())

        return self

    @jax.jit
    def reset_step(self: M) -> M:
        self.loss_and_logs.reset()
        return self

    @staticmethod
    def loss_fn(
        params: Module,
        model: "Model",
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tp.Tuple[jnp.ndarray, tp.Tuple["Model", Logs]]:
        model.module = model.module.merge(params)
        preds = model.module(x)

        loss, losses_logs, metrics_logs = model.loss_and_logs.batch_loss_epoch_logs(
            target=y,
            preds=preds,
        )
        logs = {**losses_logs, **metrics_logs}

        return loss, (model, logs)

    @jax.jit
    def train_step(
        self: M,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tp.Tuple[M, Logs]:
        print("JITTTTING")
        model = self
        params = model.module.parameters()

        grads, (model, logs) = jax.grad(model.loss_fn, has_aux=True)(
            params, model, x, y
        )

        params = model.optimizer.update(grads, params)
        model.module = model.module.merge(params)

        return model, logs

    @jax.jit
    def test_step(
        self: M,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tp.Tuple[M, Logs]:

        loss, (model, logs) = self.loss_fn(self.module, self, x, y)

        return model, logs

    @jax.jit
    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.module(x).argmax(axis=1)


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
    model = Model(
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

    model: Model = model.init_step(seed=42, inputs=X_train[:batch_size])

    print(model.module.tabulate(X_train[:batch_size], signature=True))

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
            model, train_logs = model.train_step(x, y)

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
            model, test_logs = model.test_step(x, y)

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
