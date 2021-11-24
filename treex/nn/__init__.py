from treex import types

from .batch_norm import BatchNorm
from .conv import Conv
from .dropout import Dropout
from .flatten import Flatten
from .flax_module import FlaxModule
from .linear import Linear
from .mlp import MLP
from .sequential import Lambda, Sequential, sequence

try:
    from .haiku_module import HaikuModule
except types.OptionalDependencyNotFound:
    pass
