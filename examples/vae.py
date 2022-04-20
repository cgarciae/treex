import typing as tp
from functools import partial

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

Batch = tp.Mapping[str, jnp.ndarray]
Logs = tp.Dict[str, jnp.ndarray]
np.random.seed(420)


def kl_divergence(mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(
        0.5 * jnp.mean(-jnp.log(std**2) - 1.0 + std**2 + mean**2, axis=-1)
    )


class Encoder(tx.Module):
    """Encoder model."""

    linear1: tx.Linear
    linear_mean: tx.Linear
    linear_std: tx.Linear
    kl_loss: tp.Optional[jnp.ndarray] = tx.LossLog.node()

    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
    ):

        self.linear1 = tx.Linear(hidden_size)
        self.linear_mean = tx.Linear(latent_size)
        self.linear_std = tx.Linear(latent_size)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.initializing():
            self.kl_loss = jnp.array(0.0, dtype=jnp.float32)

        assert self.kl_loss is not None

        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.linear1(x)
        x = jax.nn.relu(x)

        mean = self.linear_mean(x)
        log_std = self.linear_std(x)
        stddev = jnp.exp(log_std)

        key = self.next_key()
        z = mean + stddev * jax.random.normal(key, mean.shape)

        if self.is_mutable:
            self.kl_loss = 2e-1 * kl_divergence(mean, stddev)

        return z


class Decoder(tx.Module):
    """Decoder model."""

    linear1: tx.Linear
    linear2: tx.Linear

    def __init__(
        self,
        hidden_size: int,
        image_shape: tp.Sequence[int],
    ):

        self.linear1 = tx.Linear(hidden_size)
        self.linear2 = tx.Linear(np.prod(image_shape))
        self.output_shape = image_shape

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = self.linear1(z)
        z = jax.nn.relu(z)

        logits = self.linear2(z)
        logits = jnp.reshape(logits, (-1, *self.output_shape))

        return logits


class VAE(tx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(
        self,
        image_shape: tp.Sequence[int],
        hidden_size: int,
        latent_size: int,
    ):

        self.encoder = Encoder(hidden_size, latent_size)
        self.decoder = Decoder(hidden_size, image_shape)

    def __call__(self, x):
        return self.decoder(self.encoder(x))

    @jax.jit
    def generate(self, z):
        return jax.nn.sigmoid(self.decoder(z))

    @jax.jit
    def reconstruct(self, x):
        return jax.nn.sigmoid(self(x))


@jax.jit
def init_step(
    key: jnp.ndarray,
    model: VAE,
    optimizer: tx.Optimizer,
    metrics: jm.LossesAndMetrics,
    x: jnp.ndarray,
) -> tp.Tuple[VAE, tx.Optimizer, jm.LossesAndMetrics]:
    model = model.init(key=key)(x)
    optimizer = optimizer.init(model.trainable_parameters())
    metrics = metrics.reset(aux_losses=model.loss_logs().as_logs())
    return model, optimizer, metrics


def loss_fn(
    params: tp.Optional[VAE],
    key: tp.Optional[jnp.ndarray],
    model: VAE,
    metrics: jm.LossesAndMetrics,
    x: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, tp.Tuple[VAE, jm.LossesAndMetrics]]:

    if params is not None:
        model = model.merge(params)

    x_pred, model = model.apply(key=key)(x)

    aux_losses = model.filter(tx.LossLog).as_logs()
    loss, metrics = metrics.loss_and_update(
        target=x,
        preds=x_pred,
        aux_losses=aux_losses,
    )

    return loss, (model, metrics)


@jax.jit
def train_step(
    key: jnp.ndarray,
    model: VAE,
    optimizer: tx.Optimizer,
    metrics: jm.LossesAndMetrics,
    x: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, VAE, tx.Optimizer, jm.LossesAndMetrics]:
    params = model.trainable_parameters()
    loss_key, key = jax.random.split(key)

    grads, (model, metrics) = jax.grad(loss_fn, has_aux=True)(
        params, loss_key, model, metrics, x
    )

    params, optimizer = optimizer.update(grads, params)
    model = model.merge(params)

    key = tp.cast(jnp.ndarray, key)

    return key, model, optimizer, metrics


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 32,
    hidden_size: int = 128,
    latent_size: int = 128,
    steps_per_epoch: int = -1,
):
    image_shape = (28, 28)

    # load data
    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = np.stack(dataset["train"]["image"])
    X_test = np.stack(dataset["test"]["image"])

    X_train = (X_train > 0).astype(jnp.float32)
    X_test = (X_test > 0).astype(jnp.float32)

    model: VAE = VAE(
        image_shape=image_shape,
        hidden_size=hidden_size,
        latent_size=latent_size,
    )
    optimizer: tx.Optimizer = tx.Optimizer(optax.adamw(1e-3))
    metrics: jm.LossesAndMetrics = jm.LossesAndMetrics(
        losses=jm.losses.Crossentropy(binary=True),
        aux_losses=jm.AuxLosses(),
    )
    key = tx.Key(42)

    model, optimizer, metrics = init_step(key, model, optimizer, metrics, X_train[:32])

    print(model.tabulate(X_train[:batch_size]))

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    history: tp.List[Logs] = []

    for epoch in range(epochs):
        model = model.train()
        metrics = metrics.reset()
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
            key, model, optimizer, metrics = train_step(
                key, model, optimizer, metrics, x
            )

        logs = metrics.compute()
        history.append(logs)
        print(f"[{epoch}] {logs}")

    model = model.eval()

    # plot loss curve
    for name in history[0]:
        plt.figure()
        plt.title(name)
        plt.plot([logs[name] for logs in history])

    # visualize reconstructions
    idxs = np.random.choice(len(X_test), 10)
    x_sample = X_test[idxs]
    x_pred, model = model.apply(key=key, method="reconstruct")(x_sample)

    plt.figure()
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_sample[i], cmap="gray")
        plt.subplot(2, 5, 5 + i + 1)
        plt.imshow(x_pred[i], cmap="gray")

    # visualize samples from latent space
    z_samples = np.random.normal(size=(10, latent_size))
    samples = model.generate(z_samples)

    plt.figure()
    plt.title("Generative Samples")
    for i in range(5):
        plt.subplot(2, 5, 2 * i + 1)
        plt.imshow(samples[i], cmap="gray")
        plt.subplot(2, 5, 2 * i + 2)
        plt.imshow(samples[i + 1], cmap="gray")

    plt.show()
    plt.close()


if __name__ == "__main__":

    typer.run(main)
