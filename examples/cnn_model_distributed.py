import typing as tp
from functools import partial

import dataget
import einops
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from numpy.core.fromnumeric import reshape
from tqdm import tqdm

import treex as tx
from treex import metrics
from treex.utils import _check_rejit

Batch = tp.Mapping[str, np.ndarray]
Module = tx.Sequential
Metric = tx.metrics.Accuracy
np.random.seed(420)

M = tp.TypeVar("M", bound="Model")


class Model(tx.Module):
    module: Module
    optimizer: tx.Optimizer
    metric: Metric

    def __init__(
        self,
        module: Module,
        optimizer: optax.GradientTransformation,
        metric: Metric,
    ) -> None:
        self.module = module
        self.optimizer = tx.Optimizer(optimizer)
        self.metric = metric

    def __call__(self, *args, **kwargs) -> tp.Any:
        return self.module(*args, **kwargs)

    @partial(
        jax.pmap,
        in_axes=(None, None, 0),
        out_axes=0,
        static_broadcasted_argnums=(1,),
        axis_name="device",
    )
    def init_step(self: M, seed: int, device_idx: jnp.ndarray) -> M:
        self.module = self.module.init(seed)
        self.optimizer = self.optimizer.init(self.module.parameters())

        # assign unique rng keys
        axis_index = jax.lax.axis_index("device")
        self.map(lambda k: jax.random.fold_in(k, axis_index), tx.Rng, inplace=True)

        return self

    @partial(jax.pmap, axis_name="device")
    def reset_step(self: M) -> M:
        self.metric.reset()
        return self

    @staticmethod
    def loss_fn(
        params: Module,
        module: Module,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tp.Tuple[jnp.ndarray, tp.Tuple[Module, jnp.ndarray]]:
        module = module.merge(params)
        y_pred = module(x)

        loss = jnp.mean(
            optax.softmax_cross_entropy(
                y_pred,
                jax.nn.one_hot(y, 10),
            )
        )

        return loss, (module, y_pred)

    @partial(jax.pmap, axis_name="device")
    def train_step(
        self: M,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tp.Tuple[jnp.ndarray, M]:
        print("JITTTTING")
        params = self.module.parameters()

        (loss, (self.module, y_pred)), grads = jax.value_and_grad(
            self.loss_fn, has_aux=True
        )(params, self.module, x, y)

        assert isinstance(self.module, Module)

        params = self.optimizer.update(grads, params)
        self.module = self.module.merge(params)

        # sync batch statistics
        self.module = self.module.map(
            partial(jax.lax.pmean, axis_name="device"),
            tx.BatchStat,
        )

        # update metric
        y, y_pred = jax.lax.all_gather((y, y_pred), axis_name="device")
        y, y_pred = y.reshape(-1), y_pred.reshape(-1, y_pred.shape[-1])
        self.metric.update(y_true=y, y_pred=y_pred)

        return loss, self

    @partial(jax.pmap, axis_name="device")
    def test_step(
        self: M,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tp.Tuple[jnp.ndarray, M]:

        loss, (self.module, y_pred) = self.loss_fn(self.module, self.module, x, y)

        y, y_pred = jax.lax.all_gather((y, y_pred), axis_name="device")
        y, y_pred = y.reshape(-1), y_pred.reshape(-1, y_pred.shape[-1])
        _batch_metric = self.metric(y_true=y, y_pred=y_pred)

        return loss, self

    @jax.jit
    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.module(x).argmax(axis=1)


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

    model = Model(
        module=tx.Sequential(
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
        ),
        optimizer=optax.adamw(1e-3),
        metric=tx.metrics.Accuracy(),
    )

    model: Model = model.init_step(42, device_idx)

    # load data
    X_train, y_train, X_test, y_test = dataget.image.mnist().get()
    X_train = X_train[..., None]
    X_test = X_test[..., None]

    print(model.module.tabulate(signature=True))

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
            loss, model = model.train_step(x, y)
            train_losses.append(loss)

        epoch_train_loss = jnp.mean(jnp.stack(train_losses))
        epoch_train_losses.append(epoch_train_loss)
        epoch_train_accs.append(model.metric.compute())

        # ---------------------------------------
        # test
        # ---------------------------------------
        test_losses = []
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
            idx = np.random.choice(len(X_test), batch_size * n_devices)
            x = einops.rearrange(
                X_test[idx], "(device batch) ... -> device batch ...", device=n_devices
            )
            y = einops.rearrange(
                y_test[idx], "(device batch) ... -> device batch ...", device=n_devices
            )

            loss, model = model.test_step(x, y)
            test_losses.append(loss)

        epoch_test_loss = jnp.mean(jnp.stack(test_losses))
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_accs.append(model.metric.compute())

        print(
            f"[{epoch}] loss_train={epoch_train_loss}, acc_train={epoch_train_accs[-1]}, loss_test={epoch_test_loss}, acc_test={epoch_test_accs[-1]}"
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

    y_pred = model.predict(x_sample)

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
