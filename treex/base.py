from typing import Dict, Tuple, Type, TypeVar, Any
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np


class TreePart:
    pass


class Parameter(jnp.ndarray, TreePart):
    pass


class State(jnp.ndarray, TreePart):
    pass


@jax.tree_util.register_pytree_node_class
class Nothing:
    def tree_flatten(self):
        return (), ()

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls()


T = TypeVar("T", bound="Treex")


class Treex(TreePart):
    def _parts(self) -> Tuple[Dict[str, Tuple[Type[TreePart], Any]], Dict[str, Any]]:

        annotations = self.__class__.__annotations__
        fields = vars(self)

        tree_parts = {
            k: (cls, fields[k])
            for k, cls in annotations.items()
            if issubclass(cls, TreePart)
        }
        not_tree = {k: v for k, v in fields.items() if k not in tree_parts}

        return tree_parts, not_tree

    def tree_flatten(self):
        tree_parts, not_tree = self._parts()

        tree_parts = {k: v for k, (_cls, v) in tree_parts.items()}

        return tuple(tree_parts.values()), dict(
            tree_parts=tree_parts.keys(), not_tree=not_tree
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        module = cls.__new__(cls)

        for k, v in aux_data["not_tree"].items():
            setattr(module, k, v)

        for i, k in enumerate(aux_data["tree_parts"]):
            setattr(module, k, children[i])

        return module

    def slice(self: T, filter_cls: Type) -> T:
        module: T = self.__class__.__new__(self.__class__)
        tree_parts, not_tree = self._parts()

        for k, (cls, v) in tree_parts.items():
            if issubclass(cls, Treex):
                v = v.slice(filter_cls)
            elif not issubclass(cls, filter_cls):
                v = Nothing()

            setattr(module, k, v)

        for k, v in not_tree.items():
            setattr(module, k, v)

        return module

    def _merge_one(self: T, other: T) -> T:
        module: T = self.__class__.__new__(self.__class__)
        tree_parts, not_tree = self._parts()

        for k, (cls, v1) in tree_parts.items():
            v2 = getattr(other, k)

            if issubclass(cls, Treex):
                if isinstance(v1, Treex) and isinstance(v2, Treex):
                    v = v1._merge_one(v2)
                else:
                    v = v1 if not isinstance(v1, Nothing) else v2
            else:
                v = v1 if not isinstance(v1, Nothing) else v2

            setattr(module, k, v)

        for k, v in not_tree.items():
            setattr(module, k, v)

        return module

    def merge(self: T, other: T, *rest: T) -> T:
        others = (other,) + rest
        acc = self

        for other in others:
            acc = acc._merge_one(other)

        return acc
