from treex import types

from .conv import Conv
from .dropout import Dropout
from .embed import Embed
from .flatten import Flatten
from .flax_module import FlaxModule
from .linear import Linear
from .mlp import MLP
from .norm import BatchNorm, GroupNorm, LayerNorm
from .sequential import Lambda, Sequential, sequence

try:
    from .haiku_module import HaikuModule

    _haiku_available = True
except types.OptionalDependencyNotFound:
    _haiku_available = False

__all__ = [
    "BatchNorm",
    "Conv",
    "Dropout",
    "Embed",
    "Flatten",
    "FlaxModule",
    "GroupNorm",
    "LayerNorm",
    "Linear",
    "MLP",
    "Lambda",
    "Sequential",
    "sequence",
]

if _haiku_available:
    __all__.append("HaikuModule")
