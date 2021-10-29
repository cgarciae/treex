import threading
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass

from treex import types


@dataclass
class _Context(threading.local):
    call_info: tp.Optional[tp.Dict["Module", tp.Tuple[types.Inputs, tp.Any]]] = None

    def __enter__(self):
        global _CONTEXT
        self._old_context = _CONTEXT
        _CONTEXT = self

    def __exit__(self, *args):
        global _CONTEXT
        _CONTEXT = self._old_context

    @contextmanager
    def update(self, **kwargs):
        fields = vars(self).copy()
        fields.pop("_old_context", None)
        fields.update(kwargs)

        with _Context(**fields):
            yield


_CONTEXT = _Context()
