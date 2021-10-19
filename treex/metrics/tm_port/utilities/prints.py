import os
import warnings
from functools import wraps
from typing import Any, Callable

from treex.metrics import _logger as log


def rank_zero_only(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        if rank_zero_only.rank == 0:  # type: ignore
            return fn(*args, **kwargs)

    return wrapped_fn


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, "rank", int(os.environ.get("LOCAL_RANK", 0)))  # type: ignore


def _warn(*args: Any, **kwargs: Any) -> None:
    warnings.warn(*args, **kwargs)


def _info(*args: Any, **kwargs: Any) -> None:
    log.info(*args, **kwargs)


def _debug(*args: Any, **kwargs: Any) -> None:
    log.debug(*args, **kwargs)


rank_zero_debug = rank_zero_only(_debug)
rank_zero_info = rank_zero_only(_info)
rank_zero_warn = rank_zero_only(_warn)
