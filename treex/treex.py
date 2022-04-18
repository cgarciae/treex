import typing as tp

import treeo as to

from treex import types

A = tp.TypeVar("A")
T = tp.TypeVar("T", bound="Treex")


class Treex(
    to.Tree,
    to.Immutable,
    to.Copy,
    to.ToString,
    to.ToDict,
    to.Repr,
    to.Filter,
    to.Merge,
    to.Map,
    to.Compact,
):
    """
    A Tree class with all Mixin Extensions. Base class for all Treex classes.
    """

    def train(self: T, mode: bool = True) -> T:
        """
        Creates a new module with the same structure, but with `Module.training` set to the given value.

        Arguments:
            mode: The new training mode.
        Returns:
            The new module in with the training mode is set to the given value.
        """

        def set_training(tree):
            if isinstance(tree, Treex) and hasattr(tree, "_training"):
                tree._training = mode

        return to.apply(set_training, self)

    def eval(self: T) -> T:
        """
        Creates a new module with the training mode set to False, equivalent to calling `train(False)`.

        Returns:
            The new module with the training mode set to False.
        """
        return self.train(False)

    def freeze(self: T, mode: bool = True) -> T:
        """
        Creates a new module with the same structure, but with `Module.frozen` set
        to the given value.

        Arguments:
            mode: The new `frozen` mode.
        Returns:
            The new module in with the `frozen` mode is set to the given value.
        """

        def set_frozen(tree):
            if isinstance(tree, Treex) and hasattr(tree, "_frozen"):
                tree._frozen = mode

        return to.apply(set_frozen, self)

    def unfreeze(self: T) -> T:
        """
        Creates a new module with `.frozen` set to False, equivalent to calling `freeze(False)`.

        Returns:
            The new module with `.frozen` set to False.
        """
        return self.freeze(False)


class Filters:
    def parameters(self: A, *filters: types.Filter) -> A:
        """
        Returns a copy of the Module with only tx.Parameter TreeParts, alias for `filter(tx.Parameter)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return to.filter(self, types.Parameter, *filters)

    def trainable_parameters(self: A, *filters: types.Filter) -> A:
        """
        Returns a copy of the Module with only tx.Parameter TreeParts which are not frozen, alias for
        `filter(tx.Parameter, lambda field: not field.module.frozen)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return self.parameters(lambda field: not field.module.frozen, *filters)

    def batch_stats(self: A, *filters: types.Filter) -> A:
        """
        Returns a copy of the Module with only tx.BatchStat TreeParts, alias for `filter(tx.BatchStat)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return to.filter(self, types.BatchStat, *filters)

    def rngs(self: A, *filters: types.Filter) -> A:
        """
        Returns a copy of the Module with only tx.Rng TreeParts, alias for `filter(tx.Rng)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return to.filter(self, types.Rng, *filters)

    def model_states(self: A, *filters: types.Filter) -> A:
        """
        Returns a copy of the Module with only tx.ModelState TreeParts, alias for `filter(tx.ModelState)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return to.filter(self, types.ModelState, *filters)

    def states(self: A, *filters: types.Filter) -> A:
        """
        Returns a copy of the Module with only tx.State TreeParts, alias for `filter(tx.State)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return to.filter(self, types.State, *filters)

    def metric_logs(self: A, *filters: types.Filter) -> A:
        """
        Returns a copy of the Module with only tx.Metric TreeParts, alias for `filter(tx.Metric)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return to.filter(self, types.MetricLog, *filters)

    def loss_logs(self: A, *filters: types.Filter) -> A:
        """
        Returns a copy of the Module with only tx.Loss TreeParts, alias for `filter(tx.Loss)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return to.filter(self, types.LossLog, *filters)

    def logs(self: A, *filters: types.Filter) -> A:
        """
        Returns a copy of the Module with only tx.Log TreeParts, alias for `filter(tx.Log)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return to.filter(self, types.Log, *filters)

    def caches(self: A, *filters: types.Filter) -> A:
        """
        Returns a copy of the Module with only tx.Cache TreeParts, alias for `filter(tx.Cache)`.

        Arguments:
            filters: additional filters passed to `filter`.
        """
        return to.filter(self, types.Cache, *filters)
