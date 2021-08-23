from functools import partial
import os
import typing as tp
from datetime import datetime

import dataget
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import treex as tx
import optax

Batch = tp.Mapping[str, np.ndarray]
np.random.seed(42)


class _Loss(tx.TreePart):
    pass


Loss = tp.cast(tp.Type[jnp.ndarray], _Loss)


def kl_divergence(mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(
        0.5 * jnp.mean(-jnp.log(std ** 2) - 1.0 + std ** 2 + mean ** 2, axis=-1)
    )


class Encoder(tx.Module):
    """Encoder model."""

    linear1: tx.Linear
    linear_mean: tx.Linear
    linear_std: tx.Linear
    rng: tx.Rng
    kl_loss: Loss

    def __init__(
        self,
        image_shape: tp.Sequence[int],
        hidden_size: int,
        latent_size: int,
    ):
        self.linear1 = tx.Linear(np.prod(image_shape), hidden_size)
        self.linear_mean = tx.Linear(hidden_size, latent_size)
        self.linear_std = tx.Linear(hidden_size, latent_size)
        self.rng = tx.Initializer(lambda key: key)
        self.kl_loss = jnp.array(0.0)

    def __call__(self, x: np.ndarray) -> jnp.ndarray:
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.linear1(x)
        x = jax.nn.relu(x)

        mean = self.linear_mean(x)
        log_stddev = self.linear_std(x)
        stddev = jnp.exp(log_stddev)

        # friendly RNG interface: rng.next() == jax.random.split(...)
        assert isinstance(self.rng, jnp.ndarray)
        key, self.rng = jax.random.split(self.rng)
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
        self.encoder = Encoder(image_shape, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, image_shape)

    def __call__(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def generate(self, z):
        return jax.nn.sigmoid(self.decoder(z))

    def reconstruct(self, x):
        return jax.nn.sigmoid(self.decoder(self.encoder(x)))


@partial(jax.value_and_grad, has_aux=True)
def loss_fn(params: VAE, model: VAE, x: np.ndarray) -> tp.Tuple[jnp.ndarray, VAE]:
    model = model.update(params)
    x_pred = model(x)

    crossentropy_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(x_pred, x))
    aux_losses = jax.tree_leaves(model.filter(Loss))

    loss = crossentropy_loss + sum(aux_losses, 0.0)

    return loss, model


@jax.jit
def train_step(model: VAE, opt_state, x: np.ndarray):
    params = model.filter(tx.Parameter)
    (loss, model), grads = loss_fn(params, model, x)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    new_params = optax.apply_updates(params, updates)

    model = model.update(new_params)

    return loss, model, opt_state


X_train, _1, X_test, _2 = dataget.image.mnist(global_cache=True).get()
# Now binarize data
X_train = (X_train > 0).astype(jnp.float32)
X_test = (X_test > 0).astype(jnp.float32)

print("X_train:", X_train.shape, X_train.dtype)
print("X_test:", X_test.shape, X_test.dtype)

epochs = 20
batch_size = 32
image_shape = (28, 28)
hidden_size = 128
latent_size = 128
optimizer = optax.adam(1e-3)

model = VAE(
    image_shape=image_shape,
    hidden_size=hidden_size,
    latent_size=latent_size,
).init(42)
opt_state = optimizer.init(model.filter(tx.Parameter))


for epoch in range(epochs):
    losses = []
    model = model.train()
    for step in range(len(X_train) // batch_size):
        idx = np.random.choice(len(X_train), batch_size)
        x = X_train[idx]
        loss, model, opt_state = train_step(model, opt_state, x)
        losses.append(loss)

    print(f"[{epoch}] loss={np.mean(losses)}")

model = model.eval()

idxs = np.random.randint(0, len(X_test), size=(5,))
x_sample = X_test[idxs]
x_pred = model.reconstruct(x_sample)

# plot and save results
plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_sample[i], cmap="gray")
    plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(x_pred[i], cmap="gray")

z_samples = np.random.normal(size=(12, latent_size))
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
