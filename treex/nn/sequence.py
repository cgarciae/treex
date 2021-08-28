import typing as tp

import jax.numpy as jnp
import numpy as np

from treex.module import Module


def sequence(
    *layers: tp.Callable[[np.ndarray], np.ndarray]
) -> tp.Callable[[np.ndarray], np.ndarray]:
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

    def _sequence(x: np.ndarray) -> np.ndarray:
        for layer in layers:
            x = layer(x)
        return x

    return _sequence


CallableModule = tp.cast(
    tp.Type[tp.Callable[[np.ndarray], np.ndarray]], tp.List[Module]
)


class Sequence(Module):
    """
    A Module that applies a sequence of Modules or functions in order.

    Example:

    ```python
    mlp = tx.Sequence(
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

    layers: tp.List[CallableModule]

    def __init__(
        self, *layers: tp.Union[CallableModule, tp.Callable[[np.ndarray], np.ndarray]]
    ):
        """
        Arguments:
            *layers: A list of layers or callables to apply to apply in sequence.
        """
        self.layers = [
            layer if isinstance(layer, Module) else Lambda(layer) for layer in layers
        ]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


class Lambda(Module):
    """
    A Module that applies a pure function to its input.
    """

    f: tp.Callable[[np.ndarray], np.ndarray]

    def __init__(self, f: tp.Callable[[np.ndarray], np.ndarray]):
        """
        Arguments:
            f: A function to apply to the input.
        """
        self.f = f
        self.f = f

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Arguments:
            x: The input to the function.
        Returns:
            The output of the function.
        """
        return self.f(x)
