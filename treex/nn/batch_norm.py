import typing as tp
from flax.linen import normalization as flax_module
import jax
import jax.numpy as jnp
import numpy as np

from treex.module import Module
from treex import types


class BatchNorm(Module):
    """BatchNorm Module.

    `BatchNorm` is implemented as a wrapper over `flax.linen.BatchNorm`, its constructor
    arguments accept almost the same arguments including any Flax artifacts such as initializers.
    Main differences:

    * `use_running_average` is not a constructor argument, but remains a `__call__` argument.
    * `self.training` state is used to indicate how BatchNorm should behave, interally
    `use_running_average = not self.training` is used unless `use_running_average` is explicitly
    passed via `__call__`.
    """

    # pytree
    params: tp.Optional[tp.Mapping[str, types.Parameter]]
    batch_stats: tp.Optional[tp.Mapping[str, types.BatchStat]]

    # props
    features_in: int
    module: flax_module.BatchNorm

    def __init__(
        self,
        features_in: int,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        dtype: flax_module.Dtype = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: tp.Callable[
            [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype],
            flax_module.Array,
        ] = flax_module.initializers.zeros,
        scale_init: tp.Callable[
            [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype],
            flax_module.Array,
        ] = flax_module.initializers.ones,
        axis_name: tp.Optional[str] = None,
        axis_index_groups: tp.Any = None,
    ):
        """
        Arguments:
            features_in: the number of input features.
            use_running_average: if True, the statistics stored in batch_stats
                will be used instead of computing the batch statistics on the input.
            axis: the feature or non-batch axis of the input.
            momentum: decay rate for the exponential moving average of
                the batch statistics.
            epsilon: a small float added to variance to avoid dividing by zero.
            dtype: the dtype of the computation (default: float32).
            use_bias:  if True, bias (beta) is added.
            use_scale: if True, multiply by scale (gamma).
                When the next layer is linear (also e.g. nn.relu), this can be disabled
                since the scaling will be done by the next layer.
            bias_init: initializer for bias, by default, zero.
            scale_init: initializer for scale, by default, one.
            axis_name: the axis name used to combine batch statistics from multiple
                devices. See `jax.pmap` for a description of axis names (default: None).
            axis_index_groups: groups of axis indices within that named axis
                representing subsets of devices to reduce over (default: None). For
                example, `[[0, 1], [2, 3]]` would independently batch-normalize over
                the examples on the first two and last two devices. See `jax.lax.psum`
                for more details.
        """
        self.features_in = features_in
        self.module = flax_module.BatchNorm(
            use_running_average=None,
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            dtype=dtype,
            use_bias=use_bias,
            use_scale=use_scale,
            bias_init=bias_init,
            scale_init=scale_init,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups,
        )
        self.params = None
        self.batch_stats = None

    def module_init(self, key: jnp.ndarray):

        batch_size = 10  # random

        shape = [batch_size] * (abs(self.module.axis) + 1)  # overcompensate
        shape[self.module.axis] = self.features_in

        x = jax.random.uniform(key, shape=shape)

        variables = self.module.init(key, x, use_running_average=True)

        # Extract collections
        if "params" in variables:
            self.params = variables["params"]

        self.batch_stats = variables["batch_stats"]

    def __call__(
        self, x: np.ndarray, use_running_average: tp.Optional[bool] = None
    ) -> jnp.ndarray:
        """Normalizes the input using batch statistics.

        Arguments:
            x: the input to be normalized.
            use_running_average: if true, the statistics stored in batch_stats
                will be used instead of computing the batch statistics on the input.

        Returns:
            Normalized inputs (the same shape as inputs).
        """
        assert self.batch_stats is not None, "BatchNorm module not initialized"

        variables = dict(
            params=self.params if self.params is not None else {},
            batch_stats=self.batch_stats,
        )
        # use_running_average = True means batch_stats will not be mutated
        # self.training = True means batch_stats will be mutated
        training = (
            not use_running_average
            if use_running_average is not None
            else self.training
        )

        # call apply
        output, variables = self.module.apply(
            variables,
            x,
            mutable=["batch_stats"] if training else [],
            use_running_average=not training,
        )

        # update batch_stats
        if "batch_stats" in variables:
            self.batch_stats = variables["batch_stats"]

        return tp.cast(jnp.ndarray, output)
