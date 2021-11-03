import typing as tp

import jax.numpy as jnp
import numpy as np
import treeo as to

from treex.module import Module

CallableModule = tp.Callable[..., jnp.ndarray]


def sequence(*layers: CallableModule) -> CallableModule:
    """
    Creates a function that applies a sequence of callables to an input.

    Example:
    ```python
    class Block(tx.Module):
        linear: tx.Linear
        batch_norm: tx.BatchNorm
        dropout: tx.Dropout
        ...

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return tx.sequence(
                self.linear,
                self.batch_norm,
                self.dropout,
                jax.nn.relu,
            )(x)

    ```

    Arguments:
        *layers: A sequence of callables to apply.
    """

    def _sequence(x: jnp.ndarray) -> jnp.ndarray:
        for layer in layers:
            x = layer(x)
        return x

    return _sequence


class Sequential(Module):
    """
    A Module that applies a sequence of Modules or functions in order.

    Example:

    ```python
    mlp = tx.Sequential(
        tx.Linear(2, 32),
        jax.nn.relu,
        tx.Linear(32, 8),
        jax.nn.relu,
        tx.Linear(8, 4),
    ).init(42)

    x = np.random.uniform(size=(10, 2))
    y = mlp(x)

    assert y.shape == (10, 4)
    ```
    """

    layers: tp.List[CallableModule] = to.node()

    def __init__(
        self, *layers: tp.Union[CallableModule, tp.Callable[[jnp.ndarray], jnp.ndarray]]
    ):
        """
        Arguments:
            *layers: A list of layers or callables to apply to apply in sequence.
        """

        self.layers = [
            layer if isinstance(layer, Module) else Lambda(layer) for layer in layers
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


class Lambda(Module):
    """
    A Module that applies a pure function to its input.
    """

    f: tp.Callable[[jnp.ndarray], jnp.ndarray]

    def __init__(self, f: tp.Callable[[jnp.ndarray], jnp.ndarray]):
        """
        Arguments:
            f: A function to apply to the input.
        """

        self.f = f

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Arguments:
            x: The input to the function.
        Returns:
            The output of the function.
        """
        return self.f(x)
