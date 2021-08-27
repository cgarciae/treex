__version__ = "0.1.0"

# optimizers first
from treex.optimizer import *

# rest
from flax.linen import initializers

from treex.module import *
from treex.types import *

from . import nn
from .nn import *
