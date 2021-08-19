import typing as tp
from flax.linen import normalization as flax_module
import jax
import jax.numpy as jnp
import numpy as np

from treex.module import Module
from treex import types


class BatchNorm(Module):
    """
    A flax_module transformation applied over the last dimension of the input.

    `BatchNorm` is implemented as a wrapper over `flax.linen.BatchNorm`, its constructor
    arguments accept the same flax artifacts.
    """

    params: types.Dict[str, types.Parameter]
    batch_stats: types.Dict[str, types.State]  # should State be renamed to BatchStat?

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
        self.params = types.Initializer(
            lambda key: self._flax_init(key, features_in, axis)
        )
        self.batch_stats = None

    def post_init(self):
        assert isinstance(self.params, tp.Mapping)

        # variables was temporarily stored in params during init
        variables = self.params

        # Extract collections
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
        variables = dict(
            params=self.params,
            batch_stats=self.batch_stats,
        )
        # use_running_average = True means batch_stats will not be mutated
        # self.training = True means batch_stats will be mutated
        mutable = (
            not use_running_average
            if use_running_average is not None
            else self.training
        )

        # call apply
        output, variables = self.module.apply(
            variables,
            x,
            mutable=["batch_stats"] if mutable else [],
            use_running_average=not mutable,
        )

        # update batch_stats
        if "batch_stats" in variables:
            self.batch_stats = variables["batch_stats"]

        return tp.cast(jnp.ndarray, output)

    def _flax_init(
        self,
        key: jnp.ndarray,
        features_in,
        axis,
    ):
        batch_size = 10  # random

        shape = [batch_size] * (abs(axis) + 1)  # overcompensate
        shape[axis] = features_in

        x = jax.random.uniform(key, shape=shape)

        variables = self.module.init(key, x, use_running_average=True)

        return variables
