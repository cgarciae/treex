import enum
import typing
import typing as tp

import einops
import jax
import jax.numpy as jnp
import numpy as np


class DataType(enum.Enum):
    """Enum to represent data type.

    >>> "Binary" in list(DataType)
    True
    """

    BINARY = enum.auto()
    MULTILABEL = enum.auto()
    MULTICLASS = enum.auto()
    # MULTIDIM_MULTICLASS = enum.auto()


class AverageMethod(enum.Enum):
    """Enum to represent average method.

    >>> None in list(AverageMethod)
    True
    >>> AverageMethod.NONE == None
    True
    >>> AverageMethod.NONE == 'none'
    True
    """

    MICRO = enum.auto()
    MACRO = enum.auto()
    WEIGHTED = enum.auto()
    NONE = enum.auto()
    SAMPLES = enum.auto()


class MDMCAverageMethod(enum.Enum):
    """Enum to represent multi-dim multi-class average method."""

    GLOBAL = enum.auto()
    SAMPLEWISE = enum.auto()


def _input_squeeze(
    preds: jnp.ndarray,
    target: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
    """Remove excess dimensions."""
    if preds.shape[0] == 1:
        preds = jnp.expand_dims(preds.squeeze(), axis=0)
        target = jnp.expand_dims(target.squeeze(), axis=0)
    else:
        preds, target = preds.squeeze(), target.squeeze()
    return preds, target


def _stat_scores_update(
    preds: jnp.ndarray,
    target: jnp.ndarray,
    intended_mode: DataType,
    average_method: AverageMethod = AverageMethod.MICRO,
    mdmc_average_method: tp.Optional[MDMCAverageMethod] = None,
    num_classes: tp.Optional[int] = None,
    top_k: tp.Optional[int] = None,
    threshold: float = 0.5,
    multiclass: tp.Optional[bool] = None,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Updates and returns the the number of true positives, false positives, true negatives, false negatives.
    Raises ValueError if:

        - The `ignore_index` is not valid
        - When `ignore_index` is used with binary data
        - When inputs are multi-dimensional multi-class, and the `mdmc_average` parameter is not set

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        reduce: Defines the reduction that is applied
        mdmc_average: Defines how the multi-dimensional multi-class inputs are handeled
        num_classes: Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.
        top_k: Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class inputs
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities
        multiclass: Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be
        ignore_index: Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.
    """

    preds, target, mode = _input_format_classification(
        preds,
        target,
        mode=intended_mode,
        threshold=threshold,
        num_classes=num_classes,
        multiclass=multiclass,
        top_k=top_k,
    )

    if intended_mode != mode:
        raise ValueError(
            f"The intended mode '{intended_mode}' does not match the found mode '{mode}'."
        )

    if mode == DataType.MULTILABEL and top_k:
        raise ValueError(
            "You can not use the `top_k` parameter to calculate accuracy for multi-label inputs."
        )

    if preds.ndim == 3:
        if mdmc_average_method is None:
            raise ValueError(
                "When your inputs are multi-dimensional multi-class, you have to set the `mdmc_average` parameter"
            )
        if mdmc_average_method == MDMCAverageMethod.GLOBAL:
            preds = jnp.swapaxes(preds, 1, 2).reshape(-1, preds.shape[1])
            target = jnp.swapaxes(target, 1, 2).reshape(-1, target.shape[1])

    tp, fp, tn, fn = _stat_scores(preds, target, reduce=average_method)

    return tp, fp, tn, fn


def _stat_scores(
    preds: jnp.ndarray,
    target: jnp.ndarray,
    reduce: tp.Optional[AverageMethod] = AverageMethod.MICRO,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate the number of tp, fp, tn, fn.

    Args:
        preds:
            An ``(N, C)`` or ``(N, C, X)`` tensor of predictions (0 or 1)
        target:
            An ``(N, C)`` or ``(N, C, X)`` tensor of true target (0 or 1)
        reduce:
            One of ``'MICRO'``, ``'macro'``, ``'samples'``

    Return:
        Returns a list of 4 tensors; tp, fp, tn, fn.
        The shape of the returned tensors depnds on the shape of the inputs
        and the ``reduce`` parameter:

        If inputs are of the shape ``(N, C)``, then
        - If ``reduce='MICRO'``, the returned tensors are 1 element tensors
        - If ``reduce='macro'``, the returned tensors are ``(C,)`` tensors
        - If ``reduce'samples'``, the returned tensors are ``(N,)`` tensors

        If inputs are of the shape ``(N, C, X)``, then
        - If ``reduce='MICRO'``, the returned tensors are ``(N,)`` tensors
        - If ``reduce='macro'``, the returned tensors are ``(N,C)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,X)`` tensors
    """

    dim: typing.Union[int, typing.List[int]] = 1  # for "samples"
    if reduce == AverageMethod.MICRO:
        dim = [0, 1] if preds.ndim == 2 else [1, 2]
    elif reduce == AverageMethod.MACRO:
        dim = 0 if preds.ndim == 2 else 2

    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    tp = (true_pred * pos_pred).sum(axis=dim)
    fp = (false_pred * pos_pred).sum(axis=dim)

    tn = (true_pred * neg_pred).sum(axis=dim)
    fn = (false_pred * neg_pred).sum(axis=dim)

    return (
        tp.astype(jnp.uint32),
        fp.astype(jnp.uint32),
        tn.astype(jnp.uint32),
        fn.astype(jnp.uint32),
    )


def _input_format_classification(
    preds: jnp.ndarray,
    target: jnp.ndarray,
    mode: DataType,
    threshold: float = 0.5,
    top_k: tp.Optional[int] = None,
    num_classes: tp.Optional[int] = None,
    multiclass: tp.Optional[bool] = None,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray, DataType]:
    """Convert preds and target tensors into common format.

    Preds and targets are supposed to fall into one of these categories (and are
    validated to make sure this is the case):

    * Both preds and target are of shape ``(N,)``, and both are integers (multi-class)
    * Both preds and target are of shape ``(N,)``, and target is binary, while preds
      are a float (binary)
    * preds are of shape ``(N, C)`` and are floats, and target is of shape ``(N,)`` and
      is integer (multi-class)
    * preds and target are of shape ``(N, ...)``, target is binary and preds is a float
      (multi-label)
    * preds are of shape ``(N, C, ...)`` and are floats, target is of shape ``(N, ...)``
      and is integer (multi-dimensional multi-class)
    * preds and target are of shape ``(N, ...)`` both are integers (multi-dimensional
      multi-class)

    To avoid ambiguities, all dimensions of size 1, except the first one, are squeezed out.

    The returned output tensors will be binary tensors of the same shape, either ``(N, C)``
    of ``(N, C, X)``, the details for each case are described below. The function also returns
    a ``case`` string, which describes which of the above cases the inputs belonged to - regardless
    of whether this was "overridden" by other settings (like ``multiclass``).

    In binary case, targets are normally returned as ``(N,1)`` tensor, while preds are transformed
    into a binary tensor (elements become 1 if the probability is greater than or equal to
    ``threshold`` or 0 otherwise). If ``multiclass=True``, then then both targets are preds
    become ``(N, 2)`` tensors by a one-hot transformation; with the thresholding being applied to
    preds first.

    In multi-class case, normally both preds and targets become ``(N, C)`` binary tensors; targets
    by a one-hot transformation and preds by selecting ``top_k`` largest entries (if their original
    shape was ``(N,C)``). However, if ``multiclass=False``, then targets and preds will be
    returned as ``(N,1)`` tensor.

    In multi-label case, normally targets and preds are returned as ``(N, C)`` binary tensors, with
    preds being binarized as in the binary case. Here the ``C`` dimension is obtained by flattening
    all dimensions after the first one. However if ``multiclass=True``, then both are returned as
    ``(N, 2, C)``, by an equivalent transformation as in the binary case.

    In multi-dimensional multi-class case, normally both target and preds are returned as
    ``(N, C, X)`` tensors, with ``X`` resulting from flattening of all dimensions except ``N`` and
    ``C``. The transformations performed here are equivalent to the multi-class case. However, if
    ``multiclass=False`` (and there are up to two classes), then the data is returned as
    ``(N, X)`` binary tensors (multi-label).

    Note:
        Where a one-hot transformation needs to be performed and the number of classes
        is not implicitly given by a ``C`` dimension, the new ``C`` dimension will either be
        equal to ``num_classes``, if it is given, or the maximum label value in preds and
        target.

    Args:
        preds: jnp.ndarray with predictions (target or probabilities)
        target: jnp.ndarray with ground truth target, always integers (target)
        threshold:
            Threshold value for transforming probability/logit predictions to binary
            (0 or 1) predictions, in the case of binary or multi-label inputs.
        num_classes:
            Number of classes. If not explicitly set, the number of classes will be inferred
            either from the shape of inputs, or the maximum label in the ``target`` and ``preds``
            tensor, where applicable.
        top_k:
            Number of highest probability entries for each sample to convert to 1s - relevant
            only for (multi-dimensional) multi-class inputs with probability predictions. The
            default value (``None``) will be interepreted as 1 for these inputs.

            Should be left unset (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/overview:using the multiclass parameter>`
            for a more detailed explanation and examples.

    Returns:
        preds: binary tensor of shape ``(N, C)`` or ``(N, C, X)``
        target: binary tensor of shape ``(N, C)`` or ``(N, C, X)``
        case: The case the inputs fall in, one of ``'binary'``, ``'multi-class'``, ``'multi-label'`` or
            ``'multi-dim multi-class'``
    """
    # Remove excess dimensions
    preds, target = _input_squeeze(preds, target)

    case = _check_classification_inputs(
        preds,
        target,
        mode=mode,
        threshold=threshold,
        num_classes=num_classes,
        multiclass=multiclass,
        top_k=top_k,
    )

    if case in (DataType.BINARY, DataType.MULTILABEL) and not top_k:
        preds = (preds >= threshold).int()
        num_classes = num_classes if not multiclass else 2

    if case == DataType.MULTILABEL and top_k:
        preds = select_topk(preds, top_k)

    if case == DataType.MULTICLASS or multiclass:
        if _is_floating_point(preds):
            num_classes = preds.shape[-1]
            preds = select_topk(preds, top_k or 1)
        else:
            if num_classes is None:
                raise ValueError(
                    f"Cannot infer number of classes when preds are integers with no class dimension, please specify `num_classes`, got shape {preds.shape}"
                )

            preds = jax.nn.one_hot(preds, max(2, num_classes))

        target = jax.nn.one_hot(target, max(2, num_classes))  # type: ignore

        if multiclass is False:
            preds, target = preds[..., 1], target[..., 1]

    if (case == DataType.MULTICLASS and multiclass == True) or multiclass:
        # target = target.reshape(-1, target.shape[-2], target.shape[-1])
        # preds = preds.reshape( -1, preds.shape[-2], preds.shape[-1])
        target = einops.rearrange(target, "... N C -> (...) N C")
        preds = einops.rearrange(preds, "... N C -> (...) N C")
    else:
        # target = target.reshape(target.shape[0], -1)
        # preds = preds.reshape(preds.shape[0], -1)
        target = einops.rearrange(target, "... N -> (...) N")
        preds = einops.rearrange(preds, "... N -> (...) N")

    # Some operations above create an extra dimension for MC/binary case - this removes it
    if preds.ndim > 2:
        preds, target = preds.squeeze(0), target.squeeze(0)

    return preds.astype(jnp.int32), target.astype(jnp.int32), case


def _check_classification_inputs(
    preds: jnp.ndarray,
    target: jnp.ndarray,
    threshold: float,
    num_classes: tp.Optional[int],
    multiclass: tp.Optional[bool],
    top_k: tp.Optional[int],
    mode: DataType,
) -> DataType:
    """Performs error checking on inputs for classification.

    This ensures that preds and target take one of the shape/type combinations that are
    specified in ``_input_format_classification`` docstring. It also checks the cases of
    over-rides with ``multiclass`` by checking (for multi-class and multi-dim multi-class
    cases) that there are only up to 2 distinct target.

    In case where preds are floats (probabilities), it is checked whether they are in [0,1] interval.

    When ``num_classes`` is given, it is checked that it is consistent with input cases (binary,
    multi-label, ...), and that, if available, the implied number of classes in the ``C``
    dimension is consistent with it (as well as that max label in target is smaller than it).

    When ``num_classes`` is not specified in these cases, consistency of the highest target
    value against ``C`` dimension is checked for (multi-dimensional) multi-class cases.

    If ``top_k`` is set (not None) for inputs that do not have probability predictions (and
    are not binary), an error is raised. Similarly if ``top_k`` is set to a number that
    is higher than or equal to the ``C`` dimension of ``preds``, an error is raised.

    Preds and target tensors are expected to be squeezed already - all dimensions should be
    greater than 1, except perhaps the first one (``N``).

    Args:
        preds: jnp.ndarray with predictions (target or probabilities)
        target: jnp.ndarray with ground truth target, always integers (target)
        threshold:
            Threshold value for transforming probability/logit predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs.
        num_classes:
            Number of classes. If not explicitly set, the number of classes will be inferred
            either from the shape of inputs, or the maximum label in the ``target`` and ``preds``
            tensor, where applicable.
        top_k:
            Number of highest probability entries for each sample to convert to 1s - relevant
            only for inputs with probability predictions. The default value (``None``) will be
            interpreted as 1 for these inputs. If this parameter is set for multi-label inputs,
            it will take precedence over threshold.

            Should be left unset (``None``) for inputs with label predictions.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/overview:using the multiclass parameter>`
            for a more detailed explanation and examples.


    Return:
        case: The case the inputs fall in, one of 'binary', 'multi-class', 'multi-label' or
            'multi-dim multi-class'
    """

    # Basic validation (that does not need case/type information)
    _basic_input_validation(preds, target, threshold, multiclass)

    # Check that shape/types fall into one of the cases
    _check_shape_and_type_consistency(preds, target, mode)
    implied_classes = preds.shape[-1] if preds.shape != target.shape else None

    if (
        implied_classes is not None
        and num_classes is not None
        and implied_classes != num_classes
    ):
        raise ValueError(
            f"Number of classes in preds ({implied_classes}) and target ({num_classes}) do not match"
        )

    # Check consistency with the `C` dimension in case of multi-class data
    if preds.shape != target.shape:

        if multiclass is False and implied_classes != 2:
            raise ValueError(
                "You have set `multiclass=False`, but have more than 2 classes in your data,"
                " based on the C dimension of `preds`."
            )

    # Check that num_classes is consistent
    if num_classes:

        if mode == DataType.BINARY:
            _check_num_classes_binary(num_classes, multiclass, implied_classes)
        elif mode == DataType.MULTICLASS:
            _check_num_classes_mc(
                preds, target, num_classes, multiclass, implied_classes
            )
        elif mode.MULTILABEL:
            assert implied_classes is not None
            _check_num_classes_ml(num_classes, multiclass, implied_classes)

    # Check that top_k is consistent
    if top_k is not None:
        assert implied_classes is not None
        _check_top_k(
            top_k, mode, implied_classes, multiclass, _is_floating_point(preds)
        )

    return mode


def _basic_input_validation(
    preds: jnp.ndarray,
    target: jnp.ndarray,
    threshold: float,
    multiclass: tp.Optional[bool],
) -> None:
    """Perform basic validation of inputs that does not require deducing any information of the type of inputs."""

    if _is_floating_point(target):
        raise ValueError("The `target` has to be an integer tensor.")
    # if target.min() < 0:
    #     raise ValueError("The `target` has to be a non-negative tensor.")

    preds_float = _is_floating_point(preds)
    # if not preds_float and preds.min() < 0:
    #     raise ValueError("If `preds` are integers, they have to be non-negative.")

    if not preds.shape[0] == target.shape[0]:
        raise ValueError(
            "The `preds` and `target` should have the same first dimension."
        )

    if multiclass is False and target.max() > 1:
        raise ValueError(
            "If you set `multiclass=False`, then `target` should not exceed 1."
        )

    if multiclass is False and not preds_float and preds.max() > 1:
        raise ValueError(
            "If you set `multiclass=False` and `preds` are integers, then `preds` should not exceed 1."
        )


def _is_floating_point(x: jnp.ndarray) -> bool:
    """Check if the input is a floating point tensor."""
    return x.dtype == jnp.float16 or x.dtype == jnp.float32 or x.dtype == jnp.float64


def _check_shape_and_type_consistency(
    preds: jnp.ndarray, target: jnp.ndarray, mode: DataType
) -> None:
    """This checks that the shape and type of inputs are consistent with each other and fall into one of the
    allowed input types (see the documentation of docstring of ``_input_format_classification``). It does not check
    for consistency of number of classes, other functions take care of that.

    It returns the name of the case in which the inputs fall, and the implied number of classes (from the ``C`` dim for
    multi-class data, or extra dim(s) for multi-label data).
    """

    preds_float = _is_floating_point(preds)

    if preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )

    elif preds.ndim == target.ndim + 1:
        if mode == DataType.BINARY:
            raise ValueError(
                "If `preds` have a 1 extra dimension, then `mode` should not be `binary`."
            )

        if not preds_float:
            raise ValueError(
                "If `preds` have one dimension more than `target`, `preds` should be a float tensor."
            )
        if preds.shape[:-1] != target.shape:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be"
                " (..., C), and the shape of `target` should be (...)."
            )

    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...)"
            " and `preds` should be (N, C, ...)."
        )


def _check_num_classes_binary(
    num_classes: int, multiclass: tp.Optional[bool], implied_classes: tp.Optional[int]
) -> None:
    """This checks that the consistency of `num_classes` with the data and `multiclass` param for binary data."""

    if implied_classes is not None and implied_classes != 2:
        raise ValueError(
            "If `preds` have one dimension more than `target`, then `num_classes` should be 2 for binary data."
        )

    if num_classes > 2:
        raise ValueError("Your data is binary, but `num_classes` is larger than 2.")
    if num_classes == 2 and not multiclass:
        raise ValueError(
            "Your data is binary and `num_classes=2`, but `multiclass` is not True."
            " Set it to True if you want to transform binary data to multi-class format."
        )
    if num_classes == 1 and multiclass:
        raise ValueError(
            "You have binary data and have set `multiclass=True`, but `num_classes` is 1."
            " Either set `multiclass=None`(default) or set `num_classes=2`"
            " to transform binary data to multi-class format."
        )


def select_topk(prob_tensor: jnp.ndarray, topk: int = 1, dim: int = 1) -> jnp.ndarray:
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

    if prob_tensor.ndim > 2:
        raise NotImplementedError(
            "Support for arrays with more than 2 dimension is not yet supported"
        )

    zeros = jnp.zeros(prob_tensor.shape, dtype=jnp.uint32)
    idx_axis0 = jnp.expand_dims(jnp.arange(prob_tensor.shape[0]), axis=1)
    val, idx_axis1 = jax.lax.top_k(prob_tensor, topk)

    return zeros.at[idx_axis0, idx_axis1].set(1)


def _check_num_classes_mc(
    preds: jnp.ndarray,
    target: jnp.ndarray,
    num_classes: int,
    multiclass: tp.Optional[bool],
    implied_classes: tp.Optional[int],
) -> None:
    """This checks that the consistency of `num_classes` with the data and `multiclass` param for (multi-
    dimensional) multi-class data."""

    if num_classes == 1 and multiclass is not False:
        raise ValueError(
            "You have set `num_classes=1`, but predictions are integers."
            " If you want to convert (multi-dimensional) multi-class data with 2 classes"
            " to binary/multi-label, set `multiclass=False`."
        )
    if num_classes > 1:
        if multiclass is False and implied_classes != num_classes:
            raise ValueError(
                "You have set `multiclass=False`, but the implied number of classes "
                " (from shape of inputs) does not match `num_classes`. If you are trying to"
                " transform multi-dim multi-class data with 2 classes to multi-label, `num_classes`"
                " should be either None or the product of the size of extra dimensions (...)."
                " See Input Types in Metrics documentation."
            )
        # if num_classes <= target.max():
        #     raise ValueError(
        #         "The highest label in `target` should be smaller than `num_classes`."
        #     )
        if preds.shape != target.shape and num_classes != implied_classes:
            raise ValueError(
                "The size of C dimension of `preds` does not match `num_classes`."
            )


def _check_num_classes_ml(
    num_classes: int, multiclass: tp.Optional[bool], implied_classes: int
) -> None:
    """This checks that the consistency of `num_classes` with the data and `multiclass` param for multi-label
    data."""

    if multiclass and num_classes != 2:
        raise ValueError(
            "Your have set `multiclass=True`, but `num_classes` is not equal to 2."
            " If you are trying to transform multi-label data to 2 class multi-dimensional"
            " multi-class, you should set `num_classes` to either 2 or None."
        )
    if not multiclass and num_classes != implied_classes:
        raise ValueError(
            "The implied number of classes (from shape of inputs) does not match num_classes."
        )


def _check_top_k(
    top_k: int,
    case: DataType,
    implied_classes: int,
    multiclass: tp.Optional[bool],
    preds_float: bool,
) -> None:
    if case == DataType.BINARY:
        raise ValueError("You can not use `top_k` parameter with binary data.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("The `top_k` has to be an integer larger than 0.")
    if not preds_float:
        raise ValueError(
            "You have set `top_k`, but you do not have probability predictions."
        )
    if multiclass is False:
        raise ValueError("If you set `multiclass=False`, you can not set `top_k`.")
    if case == DataType.MULTILABEL and multiclass:
        raise ValueError(
            "If you want to transform multi-label data to 2 class multi-dimensional"
            "multi-class data using `multiclass=True`, you can not use `top_k`."
        )
    if top_k >= implied_classes:
        raise ValueError(
            "The `top_k` has to be strictly smaller than the `C` dimension of `preds`."
        )


def _accuracy_compute(
    tp: jnp.ndarray,
    fp: jnp.ndarray,
    tn: jnp.ndarray,
    fn: jnp.ndarray,
    average: tp.Optional[AverageMethod],
    mdmc_average: tp.Optional[MDMCAverageMethod],
    mode: DataType,
) -> jnp.ndarray:
    """Computes accuracy from stat scores: true positives, false positives, true negatives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives
        average: Defines the reduction that is applied.
        mdmc_average: Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter).
        mode: Mode of the input tensors

    Example:
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> threshold = 0.5
        >>> reduce = average = 'micro'
        >>> mdmc_average = 'global'
        >>> mode = _mode(preds, target, threshold, top_k=None, num_classes=None, multiclass=None)
        >>> tp, fp, tn, fn = _accuracy_update(
        ...                     preds,
        ...                     target,
        ...                     reduce,
        ...                     mdmc_average,
        ...                     threshold=0.5,
        ...                     num_classes=None,
        ...                     top_k=None,
        ...                     multiclass=None,
        ...                     ignore_index=None,
        ...                     mode=mode)
        >>> _accuracy_compute(tp, fp, tn, fn, average, mdmc_average, mode)
        tensor(0.5000)

        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> top_k, threshold = 2, 0.5
        >>> reduce = average = 'micro'
        >>> mdmc_average = 'global'
        >>> mode = _mode(preds, target, threshold, top_k, num_classes=None, multiclass=None)
        >>> tp, fp, tn, fn = _accuracy_update(
        ...                     preds,
        ...                     target,
        ...                     reduce,
        ...                     mdmc_average,
        ...                     threshold,
        ...                     num_classes=None,
        ...                     top_k=top_k,
        ...                     multiclass=None,
        ...                     ignore_index=None,
        ...                     mode=mode)
        >>> _accuracy_compute(tp, fp, tn, fn, average, mdmc_average, mode)
        tensor(0.6667)
    """

    if (
        mode == DataType.BINARY
        and average in [AverageMethod.MICRO, AverageMethod.SAMPLES]
    ) or mode == DataType.MULTILABEL:
        numerator = tp + tn
        denominator = tp + tn + fp + fn
    else:
        numerator = tp
        denominator = tp + fn

    if average == AverageMethod.MACRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        cond = tp + fp + fn == 0
        numerator = numerator[~cond]
        denominator = denominator[~cond]

    if average == AverageMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indeces = jnp.nonzero((tp | fn | fp) == 0)
        numerator[meaningless_indeces, ...] = -1
        denominator[meaningless_indeces, ...] = -1

    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != AverageMethod.WEIGHTED else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
    )


def _reduce_stat_scores(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    weights: tp.Optional[jnp.ndarray],
    average: tp.Optional[AverageMethod],
    mdmc_average: tp.Optional[MDMCAverageMethod],
    zero_division: int = 0,
) -> jnp.ndarray:
    """Reduces scores of type ``numerator/denominator`` or.

    ``weights * (numerator/denominator)``, if ``average='weighted'``.

    Args:
        numerator: A tensor with numerator numbers.
        denominator: A tensor with denominator numbers. If a denominator is
            negative, the class will be ignored (if averaging), or its score
            will be returned as ``nan`` (if ``average=None``).
            If the denominator is zero, then ``zero_division`` score will be
            used for those elements.
        weights: A tensor of weights to be used if ``average='weighted'``.
        average: The method to average the scores
        mdmc_average: The method to average the scores if inputs were multi-dimensional multi-class (MDMC)
        zero_division: The value to use for the score if denominator equals zero.
    """
    numerator, denominator = numerator.astype(jnp.float32), denominator.astype(
        jnp.float32
    )
    zero_div_mask = denominator == 0
    ignore_mask = denominator < 0

    if weights is None:
        weights_ = jnp.ones_like(denominator)
    else:
        weights_ = weights.astype(jnp.float32)

    numerator = jnp.where(
        zero_div_mask,
        jnp.array(float(zero_division)),
        numerator,
    )
    denominator = jnp.where(
        zero_div_mask | ignore_mask,
        jnp.array(1.0, dtype=denominator.dtype),
        denominator,
    )
    weights_ = jnp.where(ignore_mask, jnp.array(0.0, dtype=weights_.dtype), weights_)

    if average not in (AverageMethod.MICRO, AverageMethod.NONE, None):
        weights_ = weights_ / weights_.sum(axis=-1, keepdims=True)

    scores = weights_ * (numerator / denominator)

    # This is in case where sum(weights) = 0, which happens if we ignore the only present class with average='weighted'
    scores = jnp.where(
        jnp.isnan(scores), jnp.array(float(zero_division), dtype=scores.dtype), scores
    )

    if mdmc_average == MDMCAverageMethod.SAMPLEWISE:
        scores = scores.mean(axis=0)
        ignore_mask = ignore_mask.sum(axis=0).astype(jnp.bool_)

    if average in (AverageMethod.NONE, None):
        scores = jnp.where(
            ignore_mask, jnp.array(float("nan"), dtype=scores.dtype), scores
        )
    else:
        scores = scores.sum()

    return scores
