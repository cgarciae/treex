import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

key = jax.random.PRNGKey
_pymap = map
_pyfilter = filter
