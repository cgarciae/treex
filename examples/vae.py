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


def kl_divergence(mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(
        0.5 * jnp.mean(-jnp.log(std ** 2) - 1.0 + std ** 2 + mean ** 2, axis=-1)
    )


class Encoder(tx.Module):
    """Encoder model."""

    linear1: tx.Linear
    linear_mean: tx.Linear
    linear_std: tx.Linear
    next_key: tx.KeySeq
    kl_loss: tx.Loss[jnp.ndarray]

    def __init__(
        self,
        image_shape: tp.Sequence[int],
        hidden_size: int,
        latent_size: int,
    ):
        super().__init__()
        self.linear1 = tx.Linear(np.prod(image_shape), hidden_size)
        self.linear_mean = tx.Linear(hidden_size, latent_size)
        self.linear_std = tx.Linear(hidden_size, latent_size)
        self.next_key = tx.KeySeq()
        self.kl_loss = jnp.array(0.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.linear1(x)
        x = jax.nn.relu(x)

        mean = self.linear_mean(x)
        log_std = self.linear_std(x)
        stddev = jnp.exp(log_std)

        key = self.next_key()
        z = mean + stddev * jax.random.normal(key, mean.shape)

        self.kl_loss = 2e-1 * kl_divergence(mean, stddev)

        return z


class Decoder(tx.Module):
    """Decoder model."""

    linear1: tx.Linear
    linear2: tx.Linear

    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        image_shape: tp.Sequence[int],
    ):
        super().__init__()
        self.linear1 = tx.Linear(latent_size, hidden_size)
        self.linear2 = tx.Linear(hidden_size, np.prod(image_shape))
        self.output_shape = image_shape

    def __call__(self, z: np.ndarray) -> np.ndarray:
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
        super().__init__()
        self.encoder = Encoder(image_shape, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, image_shape)

    def __call__(self, x):
        return self.decoder(self.encoder(x))

    @jax.jit
    def generate(self, z):
        return jax.nn.sigmoid(self.decoder(z))

    @jax.jit
    def reconstruct(self, x):
        return jax.nn.sigmoid(self(x))


@partial(jax.value_and_grad, has_aux=True)
def loss_fn(params: VAE, model: VAE, x: np.ndarray) -> tp.Tuple[jnp.ndarray, VAE]:
    model = model.update(params)
    x_pred = model(x)

    crossentropy_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(x_pred, x))
    aux_losses = jax.tree_leaves(model.filter(tx.Loss))

    loss = crossentropy_loss + sum(aux_losses, 0.0)

    return loss, model


@jax.jit
def train_step(
    model: VAE, optimizer: tx.Optimizer, x: np.ndarray
) -> tp.Tuple[jnp.ndarray, VAE, tx.Optimizer]:
    params = model.filter(tx.Parameter)
    (loss, model), grads = loss_fn(params, model, x)

    params = optimizer.apply_updates(grads, params)
    model = model.update(params)

    return loss, model, optimizer


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 32,
    hidden_size: int = 128,
    latent_size: int = 128,
    steps_per_epoch: int = -1,
):
    image_shape = (28, 28)

    model = VAE(
        image_shape=image_shape,
        hidden_size=hidden_size,
        latent_size=latent_size,
    ).init(42)

    optimizer = tx.Optimizer(optax.adam(1e-3))
    optimizer = optimizer.init(model.filter(tx.Parameter))

    # load data
    X_train, _1, X_test, _2 = dataget.image.mnist().get()
    X_train = (X_train > 0).astype(jnp.float32)
    X_test = (X_test > 0).astype(jnp.float32)

    print(model.tabulate(X_train[:batch_size]))

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    epoch_losses = []
    for epoch in range(epochs):
        losses = []
        model = model.train()
        for step in range(
            len(X_train) // batch_size if steps_per_epoch < 1 else steps_per_epoch
        ):
            idx = np.random.choice(len(X_train), batch_size)
            x = X_train[idx]
            loss, model, optimizer = train_step(model, optimizer, x)
            losses.append(loss)

        epoch_loss = jnp.mean(jnp.stack(losses))
        epoch_losses.append(epoch_loss)
        print(f"[{epoch}] loss={epoch_loss}")

    model = model.eval()

    # plot loss curve
    plt.figure()
    plt.plot(epoch_losses)

    # visualize reconstructions
    idxs = np.random.choice(len(X_test), 10)
    x_sample = X_test[idxs]
    x_pred = model.reconstruct(x_sample)

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
