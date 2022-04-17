__version__ = "0.6.10"

import jax
import jax_metrics
import treeo
from flax.linen import initializers
from treeo import *

from treex.key_seq import KeySeq
from treex.module import (
    Module,
    ModuleMeta,
    compact_module,
    next_key,
    preserve_state,
    rng_key,
)
from treex.nn import *
from treex.optimizer import Optimizer
from treex.treex import Filters, Treex
from treex.types import (
    BatchStat,
    Cache,
    Inputs,
    Log,
    LossLog,
    MetricLog,
    MetricState,
    ModelState,
    Named,
    OptState,
    Parameter,
    Rng,
    State,
    TreePart,
)
from treex.utils import Key, iter_split

from . import nn

__all__ = [
    "KeySeq",
    "Module",
    "ModuleMeta",
    "compact_module",
    "preserve_state",
    "next_key",
    "rng_key",
    "Optimizer",
    "Treex",
    "Filters",
    "BatchStat",
    "Cache",
    "Log",
    "LossLog",
    "MetricLog",
    "MetricState",
    "ModelState",
    "OptState",
    "Parameter",
    "Rng",
    "State",
    "TreePart",
    "Inputs",
    "Named",
    "nn",
    "make_mutable",
]

__all__.extend(treeo.__all__)
__all__.extend(nn.__all__)
