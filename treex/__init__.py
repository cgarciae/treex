__version__ = "0.6.6"

import treeo
from flax.linen import initializers
from treeo import *

from treex.key_seq import KeySeq
from treex.losses import Loss
from treex.metrics import LossAndLogs, Metric
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
from treex.types import (
    BatchStat,
    Cache,
    Initializer,
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

from . import losses, metrics, nn, regularizers

__all__ = (
    treeo.__all__
    + nn.__all__
    + [
        "KeySeq",
        "Loss",
        "LossAndLogs",
        "Metric",
        "Module",
        "ModuleMeta",
        "compact_module",
        "preserve_state",
        "next_key",
        "rng_key",
        "Optimizer",
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
        "Initializer",
        "Inputs",
        "Named",
        "losses",
        "metrics",
        "nn",
        "regularizers",
    ]
)
