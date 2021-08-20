import typing as tp
import numpy as np


def sequence(
    *layers: tp.Callable[[np.ndarray], np.ndarray]
) -> tp.Callable[[np.ndarray], np.ndarray]:
    def _sequence(x: np.ndarray) -> np.ndarray:
        for layer in layers:
            x = layer(x)
        return x

    return _sequence
