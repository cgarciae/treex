from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax._src.lax.lax import top_k

Tensor = jnp.ndarray
tensor = jnp.array

from treex.metrics.tm_port.utilities.prints import rank_zero_warn

METRIC_EPS = 1e-6


def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    x = x if isinstance(x, (list, tuple)) else [x]
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)


def dim_zero_sum(x: Tensor) -> Tensor:
    return torch.sum(x, dim=0)


def dim_zero_mean(x: Tensor) -> Tensor:
    return torch.mean(x, dim=0)


def _flatten(x: Sequence) -> list:
    return [item for sublist in x for item in sublist]


to_onehot = jax.nn.one_hot


# def to_onehot(
#     label_tensor: Tensor,
#     num_classes: Optional[int] = None,
# ) -> Tensor:
#     """Converts a dense label tensor to one-hot format.

#     Args:
#         label_tensor: dense label tensor, with shape [N, d1, d2, ...]
#         num_classes: number of classes C

#     Returns:
#         A sparse label tensor with shape [N, C, d1, d2, ...]

#     Example:
#         >>> x = torch.tensor([1, 2, 3])
#         >>> to_onehot(x)
#         tensor([[0, 1, 0, 0],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 1]])
#     """
#     if num_classes is None:
#         num_classes = int(label_tensor.max().detach().item() + 1)

#     tensor_onehot = jnp.zeros(
#         (label_tensor.shape[0], num_classes, *label_tensor.shape[1:]),
#         dtype=label_tensor.dtype,
#         # device=label_tensor.device,
#     )
#     index = jnp.broadcast_to(
#         label_tensor.astype(jnp.uint32)[:None], tensor_onehot.shape
#     )
#     return tensor_onehot.scatter_(1, index, 1.0)


def select_topk(prob_tensor: Tensor, topk: int = 1, dim: int = 1) -> Tensor:
    """Convert a probability tensor to binary by selecting top-k highest entries.

    Args:
        prob_tensor: dense tensor of shape ``[..., C, ...]``, where ``C`` is in the
            position defined by the ``dim`` argument
        topk: number of highest entries to turn into 1s
        dim: dimension on which to compare entries

    Returns:
        A binary tensor of the same shape as the input tensor of type torch.int32

    Example:
        >>> x = torch.tensor([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
        >>> select_topk(x, topk=2)
        tensor([[0, 1, 1],
                [1, 1, 0]], dtype=torch.int32)
    """
    # if topk == 1:  # argmax has better performance than topk
    #     topk_tensor = zeros.scatter(
    #         dim, jnp.expand_dims(jnp.argmax(prob_tensor, axis=dim), axis=dim), 1.0
    #     )
    # else:
    #     topk_tensor = zeros.scatter(dim, prob_tensor.topk(k=topk, dim=dim).indices, 1.0)

    if prob_tensor.ndim > 2:
        raise NotImplementedError(
            "Support for arrays with more than 2 dimension is not yet supported"
        )

    zeros = jnp.zeros(prob_tensor.shape, dtype=jnp.uint32)
    idx_axis0 = jnp.expand_dims(jnp.arange(prob_tensor.shape[0]), axis=1)
    val, idx_axis1 = jax.lax.top_k(prob_tensor, topk)

    # tops = jax.nn.one_hot(idx, prob_tensor.shape[-1], dtype=jnp.int32)
    # jnp.take_along_axis
    return jax.ops.index_update(zeros, jax.ops.index[idx_axis0, idx_axis1], 1)


def to_categorical(x: Tensor, argmax_dim: int = 1) -> Tensor:
    """Converts a tensor of probabilities to a dense label tensor.

    Args:
        x: probabilities to get the categorical label [N, d1, d2, ...]
        argmax_dim: dimension to apply

    Return:
        A tensor with categorical target [N, d2, ...]

    Example:
        >>> x = torch.tensor([[0.2, 0.5], [0.9, 0.1]])
        >>> to_categorical(x)
        tensor([1, 0])
    """
    return torch.argmax(x, dim=argmax_dim)


def get_num_classes(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
) -> int:
    """Calculates the number of classes for a given prediction and target tensor.

    Args:
        preds: predicted values
        target: true target
        num_classes: number of classes if known

    Return:
        An integer that represents the number of classes.
    """
    num_target_classes = int(target.max().detach().item() + 1)
    num_pred_classes = int(preds.max().detach().item() + 1)
    num_all_classes = max(num_target_classes, num_pred_classes)

    if num_classes is None:
        num_classes = num_all_classes
    elif num_classes != num_all_classes:
        rank_zero_warn(
            f"You have set {num_classes} number of classes which is"
            f" different from predicted ({num_pred_classes}) and"
            f" target ({num_target_classes}) number of classes",
            RuntimeWarning,
        )
    return num_classes


def apply_to_collection(
    data: Any,
    dtype: Union[type, tuple],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections is of
            the :attr:`wrong_type` even if it is of type :attr`dtype`
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        the resulting collection

    Example:
        >>> apply_to_collection(torch.tensor([8, 0, 2, 6, 7]), dtype=Tensor, function=lambda x: x ** 2)
        tensor([64,  0,  4, 36, 49])
        >>> apply_to_collection([8, 0, 2, 6, 7], dtype=int, function=lambda x: x ** 2)
        [64, 0, 4, 36, 49]
        >>> apply_to_collection(dict(abc=123), dtype=int, function=lambda x: x ** 2)
        {'abc': 15129}
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype) and (
        wrong_dtype is None or not isinstance(data, wrong_dtype)
    ):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        return elem_type(
            {
                k: apply_to_collection(v, dtype, function, *args, **kwargs)
                for k, v in data.items()
            }
        )

    if isinstance(data, tuple) and hasattr(data, "_fields"):  # named tuple
        return elem_type(
            *(apply_to_collection(d, dtype, function, *args, **kwargs) for d in data)
        )

    if isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type(
            [apply_to_collection(d, dtype, function, *args, **kwargs) for d in data]
        )

    # data is neither of dtype, nor a collection
    return data


def get_group_indexes(indexes: Tensor) -> List[Tensor]:
    """Given an integer `torch.Tensor` `indexes`, return a `torch.Tensor` of indexes for each different value in
    `indexes`.

    Args:
        indexes: a `torch.Tensor`

    Return:
        A list of integer `torch.Tensor`s

    Example:
        >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        >>> get_group_indexes(indexes)
        [tensor([0, 1, 2]), tensor([3, 4, 5, 6])]
    """

    res: dict = {}
    for i, _id in enumerate(indexes):
        _id = _id.item()
        if _id in res:
            res[_id] += [i]
        else:
            res[_id] = [i]

    return [tensor(x, dtype=torch.long) for x in res.values()]
