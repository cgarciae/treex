import typing as tp

import treeo as to

from treex import types

T = tp.TypeVar("T", bound="Treex")
Filter = tp.Union[
    tp.Type[tp.Type[tp.Any]],
    tp.Callable[[to.FieldInfo], bool],
]


class Treex(to.Tree, to.Extensions):
    """
    A Tree class with all Mixin Extensions. Base class for all Treex classes.
    """

    pass
