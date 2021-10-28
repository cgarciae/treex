import dataclasses
import functools
import inspect
import io
import re
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import treeo as to
import yaml
from jax._src.numpy.lax_numpy import split
from rich.console import Console
from treeo.utils import _get_name, _lower_snake_case, _unique_name, _unique_names

from treex import types

_pymap = map
_pyfilter = filter

LEAF_TYPES = (to.Nothing, types.Initializer, type(None))
PAD = r"{pad}"


# --------------------------------------------------------------------
# public
# --------------------------------------------------------------------


def iter_split(key: tp.Any, num: int = 2) -> tp.Tuple[tp.Any, ...]:
    splits = jax.random.split(key, num)
    return tuple(splits[i] for i in range(num))


def Key(seed: tp.Union[int, jnp.ndarray]) -> jnp.ndarray:
    return jax.random.PRNGKey(seed) if isinstance(seed, int) else seed


# --------------------------------------------------------------------
# private
# --------------------------------------------------------------------


def _get_rich_repr(table):
    f = io.StringIO()
    console = Console(file=f, force_terminal=True)
    console.print(table)

    return f.getvalue()


def _get_repr(
    obj, level: int, array_type: tp.Optional[type], inline, space="  "
) -> str:
    indent_level = space * level

    if isinstance(obj, to.Tree):

        (tree,), _ = obj.tree_flatten()

        params = {}
        submodules = {}

        for field, value in tree.items():
            tree_types = jax.tree_flatten(
                getattr(obj, field), is_leaf=lambda x: isinstance(x, to.Tree)
            )[0]
            tree_types = [type(x) for x in tree_types if isinstance(x, to.Tree)]

            annotation: tp.Union[
                tp.Type[to.Tree], tp.Type[types.TreePart], None
            ] = _first_issubclass(obj.field_metadata[field].kind, types.TreePart) or (
                to.Tree
                if len(tree_types) > 1
                else tree_types[0]
                if len(tree_types) == 1
                else None
            )

            if annotation is None:
                continue
            if _generic_issubclass(annotation, to.Tree):
                submodules[field] = value
            elif _generic_issubclass(annotation, types.TreePart):
                params[field] = value
            else:
                continue

        body = [
            indent_level
            + space
            + f"{field}: {_get_repr(value, level + 1, obj.field_metadata[field].kind, inline=True)}"
            for field, value in params.items()
        ] + [
            indent_level
            + space
            + f"{field}: {_get_repr(value, level + 1, obj.field_metadata[field].kind, inline=True)}"
            for field, value in submodules.items()
        ]

        body_str = "\n".join(body)
        end_dot = ":" if not inline else ""
        type_str = (
            f"[dim]{obj.__class__.__name__}[/]" if inline else obj.__class__.__name__
        )
        # signature_repr = _format_module_signature(obj, not_tree)

        return f"{type_str}{end_dot}\n{body_str}"

    elif isinstance(obj, tp.Mapping):
        body = [
            indent_level
            + space
            + f"{field}: {_get_repr(value, level + 1, array_type, inline=True)}"
            for field, value in obj.items()
        ]
        body_str = "\n".join(body)
        end_dot = ":" if not inline else ""
        type_str = (
            f"[dim]{obj.__class__.__name__}[/]" if inline else obj.__class__.__name__
        )
        return f"{type_str}{end_dot}\n{body_str}"
        # return f"\n{body_str}"
    elif isinstance(obj, tp.Sequence):
        body = [
            indent_level
            + space
            + f"- {_get_repr(value, level + 2, array_type, inline=False)}"
            for i, value in enumerate(obj)
        ]
        body_str = "\n".join(body)
        end_dot = ":" if not inline else ""
        type_str = (
            f"[dim]{obj.__class__.__name__}[/]" if inline else obj.__class__.__name__
        )
        return f"{type_str}{end_dot}\n{body_str}"

    elif isinstance(obj, (np.ndarray, jnp.ndarray)):
        assert array_type is not None

        type_name = array_type.__name__
        if type_name.startswith("_"):
            type_name = type_name[1:]

        shape = ", ".join(str(x) for x in obj.shape)
        return f"{type_name}([green]{shape}[/]) [dim]{obj.dtype}[/]"
    else:
        return repr(obj)


def _get_tabulate_rows(
    path,
    obj,
    level,
    tree_part_types,
    include_signature,
    include_param_type,
) -> tp.Iterable[tp.List[tp.Any]]:
    if isinstance(obj, to.Tree):
        (tree,), not_tree = obj.tree_flatten()

        params = {}
        submodules = {}

        for field, value in tree.items():
            tree_types = jax.tree_flatten(
                getattr(obj, field), is_leaf=lambda x: isinstance(x, to.Tree)
            )[0]
            tree_types = [type(x) for x in tree_types if isinstance(x, to.Tree)]

            annotation: tp.Union[
                tp.Type[to.Tree], tp.Type[types.TreePart], None
            ] = _first_issubclass(obj.field_metadata[field].kind, types.TreePart) or (
                to.Tree
                if len(tree_types) > 1
                else tree_types[0]
                if len(tree_types) == 1
                else None
            )

            if annotation is None:
                continue
            if issubclass(annotation, to.Tree):
                submodules[field] = value
            elif issubclass(annotation, types.TreePart):
                params[field] = value
            else:
                continue

        path_str = "".join(str(x) for x in path)
        signature = _format_module_signature(obj) if include_signature else {}
        argumentes_str = ", ".join(
            f"{arg} = {value}" for arg, value in signature.items()
        )
        signature_str = f"{obj.__class__.__name__}({argumentes_str})"

        if len(signature_str) > 40:
            argumentes_str = ",\n  ".join(
                f"{arg} = {value}" for arg, value in signature.items()
            )
            signature_str = f"{obj.__class__.__name__}(\n  {argumentes_str}\n)"

        params_repr = {
            field: jax.tree_map(
                lambda x: str(x)
                if isinstance(x, LEAF_TYPES)
                else _format_param(
                    x, obj.field_metadata[field].kind, include_param_type
                ),
                value,
                is_leaf=lambda x: isinstance(x, LEAF_TYPES),
            )
            for field, value in params.items()
        }
        params_repr = _simplify(params_repr)
        params_str = _as_yaml_str(params_repr)

        if level == 0:
            tree_part_sizes = [
                _format_obj_size(obj.filter(t), add_padding=True)
                for t in tree_part_types
            ]
        else:
            tree_part_sizes = [
                _format_obj_size(
                    [
                        value
                        for field, value in params.items()
                        if obj.field_metadata[field].kind is t
                    ],
                    add_padding=True,
                )
                for t in tree_part_types
            ]

        yield [obj, path_str, signature_str, params_str] + tree_part_sizes

        if level != 0:
            for field, value in submodules.items():
                yield from _get_tabulate_rows(
                    path + (f".{field}",),
                    value,
                    level - 1,
                    tree_part_types,
                    include_signature,
                    include_param_type,
                )
    elif isinstance(obj, tp.Mapping):
        for field, value in obj.items():
            if isinstance(field, str):
                field = f'"{field}"'

            yield from _get_tabulate_rows(
                path + (f"[{field}]",),
                value,
                level,
                tree_part_types,
                include_signature,
                include_param_type,
            )
    elif isinstance(obj, tp.Sequence):
        for i, value in enumerate(obj):
            yield from _get_tabulate_rows(
                path + (f"[{i}]",),
                value,
                level,
                tree_part_types,
                include_signature,
                include_param_type,
            )
    else:
        raise ValueError(f"Unsupported type {type(obj)}")


def _as_yaml_str(value) -> str:
    if hasattr(value, "__len__") and len(value) == 0:
        return ""

    file = io.StringIO()
    yaml.safe_dump(
        value,
        file,
        default_flow_style=False,
        indent=2,
        sort_keys=False,
        explicit_end=False,
    )
    return file.getvalue().replace("\n...", "").replace("'", "")


def _simplify(obj):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, tp.Sequence):
        return [_simplify(x) for x in obj]
    elif isinstance(obj, tp.Mapping):
        return {k: _simplify(v) for k, v in obj.items()}
    else:
        return obj


def _format_module_signature(obj: to.Tree) -> tp.Dict[str, str]:
    signature = {}
    simple_types = {int, str, float, bool}
    cls = obj.__class__
    values = list(inspect.signature(cls.__init__).parameters.values())[1:]
    names = {x.name for x in values}

    for field, value in vars(obj).items():
        if field in names:
            all_types = set(jax.tree_leaves(jax.tree_map(type, value)))

            if all_types.issubset(simple_types):
                signature[field] = value
            elif isinstance(value, to.ArrayLike) and value.shape == ():
                signature[field] = value

    signature = jax.tree_map(
        lambda x: f'"{x}"'
        if isinstance(x, str)
        else (
            f"{x:.2e}"
            if x < 1e-2
            else f"{x:.3f}"
            if x < 1
            else f"{x:.2f}"
            if x < 100
            else f"{x:.1f}"
            if x < 1000
            else f"{x:.2e}"
        )
        if isinstance(x, float)
        else str(x),
        signature,
    )
    signature = {
        field: str(value).replace("'", "") for field, value in signature.items()
    }

    return signature


def _format_param(obj, array_type, include_param_type) -> str:

    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        if include_param_type:
            type_name = array_type.__name__
            if type_name.startswith("_"):
                type_name = type_name[1:]
        else:
            type_name = ""

        shape = ", ".join(str(x) for x in obj.shape)
        return f"{type_name}([green]{shape}[/]){PAD}  [dim]{obj.dtype}[/]"
    else:
        return str(obj) if include_param_type else f"() [dim]{type(obj).__name__}[/]"


def _format_param_tree(params) -> str:

    params_repr = jax.tree_map(
        lambda x: _format_param(x, None, include_param_type=False),
        params,
        is_leaf=lambda x: isinstance(x, to.Nothing),
    )

    params_repr = _simplify(params_repr)
    params_str = _as_yaml_str(params_repr)

    return params_str


def _format_obj_size(obj, add_padding):
    count = sum((x.size if hasattr(x, "size") else 0 for x in jax.tree_leaves(obj)), 0)
    size = sum(
        (x.nbytes if hasattr(x, "nbytes") else 0 for x in jax.tree_leaves(obj)), 0
    )
    padding = f"{PAD}" if add_padding else ""

    return (
        f"[green]{count:,}[/]{padding}  [dim]{_format_size(size)}[/]"
        if count > 0
        else ""
    )


def _add_padding(rows):
    n_cols = len(rows[0])

    for col in range(n_cols):
        max_length = max(
            len(line.split(PAD)[0]) for row in rows for line in row[col].split("\n")
        )

        for row in rows:
            if PAD in row[col]:
                row[col] = "\n".join(
                    line.format(
                        pad=" " * (max_length - len(line.rstrip().split(PAD)[0]))
                    )
                    if PAD in line
                    else line
                    for line in row[col].rstrip().split("\n")
                )


def _format_size(size):
    count, units = (
        (f"{size / 1e9 :,.1f}", "GB")
        if size > 1e9
        else (f"{size / 1e6 :,.1f}", "MB")
        if size > 1e6
        else (f"{size / 1e3 :,.1f}", "KB")
        if size > 1e3
        else (f"{size:,}", "B")
    )

    return f"{count}{units}"


def _safe_issubclass(a, b) -> bool:
    return issubclass(a, b) if isinstance(a, tp.Type) else issubclass(type(a), b)


def _generic_issubclass(__cls, __class_or_tuple) -> bool:
    return _safe_issubclass(__cls, __class_or_tuple) or (
        hasattr(__cls, "__origin__")
        and _safe_issubclass(__cls.__origin__, __class_or_tuple)
    )


def _first_issubclass(cls: type, __class_or_tuple) -> tp.Optional[tp.Type[tp.Any]]:
    for t in to.utils._all_types(cls):
        if issubclass(t, __class_or_tuple):
            return t


def _check_rejit(f):

    cache_args = None
    cache_kwargs = None

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        nonlocal cache_args, cache_kwargs

        if cache_args is None or cache_kwargs is None:
            cache_args = args
            cache_kwargs = kwargs
        else:
            jax.tree_map(lambda *xs: None, args, cache_args)
            jax.tree_map(lambda *xs: None, kwargs, cache_kwargs)

        return f(*args, **kwargs)

    return wrapper


def _flatten_names(inputs: tp.Any) -> tp.List[tp.Tuple[str, tp.Any]]:
    return [
        ("/".join(map(str, path)), value)
        for path, value in _flatten_names_helper((), inputs)
    ]


def _flatten_names_helper(
    path: types.PathLike, inputs: tp.Any
) -> tp.Iterable[tp.Tuple[types.PathLike, tp.Any]]:

    if isinstance(inputs, (tp.Tuple, tp.List)):
        for i, value in enumerate(inputs):
            yield from _flatten_names_helper(path, value)
    elif isinstance(inputs, tp.Dict):
        for name, value in inputs.items():
            yield from _flatten_names_helper(path + (name,), value)
    else:
        yield (path, inputs)


def _function_argument_names(f) -> tp.Optional[tp.List[str]]:
    """
    Returns:
        A list of keyword argument names or None if variable keyword arguments (`**kwargs`) are present.
    """
    kwarg_names = []

    for k, v in inspect.signature(f).parameters.items():
        if v.kind == inspect.Parameter.VAR_KEYWORD:
            return None

        kwarg_names.append(k)

    return kwarg_names


def _maybe_expand_dims(
    a: jnp.ndarray, b: jnp.ndarray
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
    assert np.prod(a.shape) == np.prod(b.shape)

    if a.ndim < b.ndim:
        a = a[..., None]

    if b.ndim < a.ndim:
        b = b[..., None]

    return a, b
