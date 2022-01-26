import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import treeo as to
from flax.linen import normalization as flax_module

from treex import types, utils
from treex.module import Module, next_key


class BatchNorm(Module):
    """BatchNorm Module.

    `BatchNorm` is implemented as a wrapper over `flax.linen.BatchNorm`, its constructor
    arguments accept almost the same arguments including any Flax artifacts such as initializers.
    Main differences:

    * `use_running_average` is not a constructor argument, but remains a `__call__` argument.
    * `self.training` state is used to indicate how BatchNorm should behave, interally
    `use_running_average = not self.training or self.frozen` is used unless `use_running_average` is explicitly
    passed via `__call__`.
    """

    # pytree
    mean: tp.Optional[jnp.ndarray] = types.BatchStat.node()
    var: tp.Optional[jnp.ndarray] = types.BatchStat.node()
    scale: tp.Optional[jnp.ndarray] = types.Parameter.node()
    bias: tp.Optional[jnp.ndarray] = types.Parameter.node()
    momentum: jnp.ndarray = to.node()

    # props
    axis: int
    epsilon: float
    dtype: flax_module.Dtype
    use_bias: bool
    use_scale: bool
    bias_init: tp.Callable[
        [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype],
        flax_module.Array,
    ]
    scale_init: tp.Callable[
        [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype],
        flax_module.Array,
    ]
    axis_name: tp.Optional[str]
    axis_index_groups: tp.Any

    def __init__(
        self,
        *,
        axis: int = -1,
        momentum: tp.Union[float, jnp.ndarray] = 0.99,
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

        self.axis = axis
        self.momentum = jnp.asarray(momentum)
        self.epsilon = epsilon
        self.dtype = dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups

        self.mean = None
        self.var = None
        self.scale = None
        self.bias = None

    @property
    def module(self) -> flax_module.BatchNorm:
        return flax_module.BatchNorm(
            use_running_average=None,
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            dtype=self.dtype,
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            bias_init=self.bias_init,
            scale_init=self.scale_init,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
        )

    def __call__(
        self, x: jnp.ndarray, use_running_average: tp.Optional[bool] = None
    ) -> jnp.ndarray:
        """Normalizes the input using batch statistics.

        Arguments:
            x: the input to be normalized.
            use_running_average: if true, the statistics stored in batch_stats
                will be used instead of computing the batch statistics on the input.

        Returns:
            Normalized inputs (the same shape as inputs).
        """
        if self.initializing():
            variables = self.module.init(
                next_key(),
                x,
                use_running_average=True,
            ).unfreeze()

            # Extract collections
            if "params" in variables:
                params = variables["params"]

                if self.use_bias:
                    self.bias = params["bias"]

                if self.use_scale:
                    self.scale = params["scale"]

            self.mean = variables["batch_stats"]["mean"]
            self.var = variables["batch_stats"]["var"]

        params = {}

        if self.use_bias:
            params["bias"] = self.bias

        if self.use_scale:
            params["scale"] = self.scale

        variables = dict(
            batch_stats=dict(
                mean=self.mean,
                var=self.var,
            ),
            params=params,
        )
        # use_running_average = True means batch_stats will not be mutated
        # self.training = True means batch_stats will be mutated
        training = (
            not use_running_average
            if use_running_average is not None
            else self.training and not self.frozen and self.initialized
        )

        # call apply
        output, variables = self.module.apply(
            variables,
            x,
            mutable=["batch_stats"] if training else [],
            use_running_average=not training,
        )
        variables = variables.unfreeze()

        # update batch_stats
        if "batch_stats" in variables:
            self.mean = variables["batch_stats"]["mean"]
            self.var = variables["batch_stats"]["var"]

        return tp.cast(jnp.ndarray, output)
