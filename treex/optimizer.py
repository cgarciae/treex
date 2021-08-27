import functools
import inspect
import typing as tp

import jax
import jax.numpy as jnp
import optax
from treex.module import TreeObject
from treex.types import _State
import chex

T = tp.TypeVar("T", bound="Optimizer")
A = tp.TypeVar("A", bound="tp.Any")
C = tp.TypeVar("C")


class _OptState(_State):
    pass


OptState = tp.cast(tp.Type[tp.Any], _OptState)


class Optimizer(TreeObject):
    opt_state: tp.Optional[OptState]
    optimizer: optax.GradientTransformation

    _initialized: bool = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _get_props(self) -> tp.Dict[str, tp.Any]:
        return dict(_intialized=self._initialized)

    def __init__(self, optimizer: optax.GradientTransformation):
        self.opt_state = None
        self.optimizer = optimizer

    def init(self: T, state: tp.Any) -> T:
        module = self.copy()
        module.opt_state = module.optimizer.init(state)
        module._initialized = True
        return module

    def update(
        self, grads: A, params: tp.Optional[A] = None, apply_updates: bool = True
    ) -> A:
        assert self.opt_state is not None
        if apply_updates and params is None:
            raise ValueError("params must be provided to apply update")

        param_updates: A
        param_updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, params
        )

        if apply_updates:
            return optax.apply_updates(params, param_updates)

        return param_updates

    # THE FOLOWING METHODS ARE AUTOMATICALLY GENERATED
    # >>> DO NOT MODIFY <<<
    # run `python -m treex.optimizer` to update methods
    # --------------------------------------------------------------------------
    # <<<CODEGEN START>>>

    @staticmethod
    def adabelief(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-08,
    ) -> "Optimizer":
        r"""The AdaBelief optimiser.

        AdaBelief is an adaptive learning rate optimiser that focuses on fast
        convergence, generalisation, and stability. It adapts the step size depending
        on its "belief" in the gradient direction — the optimiser adaptively scales
        the step size by the difference between the predicted and observed gradients.
        AdaBelief is a modified version of Adam and contains the same number of
        parameters.

        References:
          Zhuang et al, 2020: https://arxiv.org/abs/2010.07468

        Args:
          learning_rate: this is a fixed global scaling factor.
          b1: the exponential decay rate to track the first moment of past gradients.
          b2: the exponential decay rate to track the second moment of past gradients.
          eps: a small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(
            optax.adabelief(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
        )

    @staticmethod
    def adafactor(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
            None,
        ] = None,
        min_dim_size_to_factor: int = 128,
        decay_rate: float = 0.8,
        decay_offset: int = 0,
        multiply_by_parameter_scale: float = True,
        clipping_threshold: tp.Union[float, None] = 1.0,
        momentum: tp.Union[float, None] = None,
        dtype_momentum: tp.Any = jnp.float32,
        weight_decay_rate: tp.Union[float, None] = None,
        eps: float = 1e-30,
        factored: bool = True,
    ) -> "Optimizer":
        r"""The Adafactor optimiser.

        Adafactor is an adaptive learning rate optimiser that focuses on fast
        training of large scale neural networks. It saves memory by using a factored
        estimate of the second order moments used to scale gradients.

        References:
          Zhuang et al, 2020: https://arxiv.org/abs/2010.07468

        Args:
            learning_rate: (float) a step size. Note: the natural scale for
              Adafactor's LR is markedly different from Adam, one doesn't use the
              1/sqrt(hidden) correction for this optim with attention-based models.
            min_dim_size_to_factor: (int) only factor the statistics if two array
              dimensions have at least this size.
            decay_rate: (float) controls second-moment exponential decay schedule.
            decay_offset: (int) for finetuning, one may set this to the starting
              step number of the finetuning phase.
            multiply_by_parameter_scale: (bool): if True, then scale learning_rate by
              parameter norm. if False, provided learning_rate is absolute step size.
            clipping_threshold: (float>=1) optional value; if None, clipping disabled.
            momentum: (float) optional value between 0 and 1, enables
              momentum and uses extra memory if non-None! None by default.
            dtype_momentum: (dtype) dtype of momentum buffers.
            weight_decay_rate: (float) optional rate at which to decay weights.
            eps: (float) regularization constant for root mean squared gradient.
            factored: (bool) whether to use factored second-moment estimates.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(
            optax.adafactor(
                learning_rate=learning_rate,
                min_dim_size_to_factor=min_dim_size_to_factor,
                decay_rate=decay_rate,
                decay_offset=decay_offset,
                multiply_by_parameter_scale=multiply_by_parameter_scale,
                clipping_threshold=clipping_threshold,
                momentum=momentum,
                dtype_momentum=dtype_momentum,
                weight_decay_rate=weight_decay_rate,
                eps=eps,
                factored=factored,
            )
        )

    @staticmethod
    def adagrad(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        initial_accumulator_value: float = 0.1,
        eps: float = 1e-07,
    ) -> "Optimizer":
        r"""The Adagrad optimizer.

        Adagrad is an algorithm for gradient based optimisation that anneals the
        learning rate for each parameter during the course of training.

        WARNING: Adagrad's main limit is the monotonic accumulation of squared
        gradients in the denominator: since all terms are >0, the sum keeps growing
        during training and the learning rate eventually becomes vanishingly small.

        References:
          Duchi et al, 2011: https://jmlr.org/papers/v12/duchi11a.html

        Args:
          learning_rate: this is a fixed global scaling factor.
          initial_accumulator_value: initialisation for the accumulator.
          eps: a small constant applied to denominator inside of the square root
            (as in RMSProp) to avoid dividing by zero when rescaling.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(
            optax.adagrad(
                learning_rate=learning_rate,
                initial_accumulator_value=initial_accumulator_value,
                eps=eps,
            )
        )

    @staticmethod
    def adam(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-08,
        eps_root: float = 0.0,
    ) -> "Optimizer":
        r"""The classic Adam optimiser.

        Adam is an SGD variant with learning rate adaptation. The `learning_rate`
        used for each weight is computed from estimates of first- and second-order
        moments of the gradients (using suitable exponential moving averages).

        References:
          Kingma et al, 2014: https://arxiv.org/abs/1412.6980

        Args:
          learning_rate: this is a fixed global scaling factor.
          b1: the exponential decay rate to track the first moment of past gradients.
          b2: the exponential decay rate to track the second moment of past gradients.
          eps: a small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
          eps_root: (default `0`), a small constant applied to denominator inside the
            square root (as in RMSProp), to avoid dividing by zero when rescaling.
            This is needed for example when computing (meta-)gradients through Adam.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(
            optax.adam(
                learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, eps_root=eps_root
            )
        )

    @staticmethod
    def adamw(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-08,
        eps_root: float = 0.0,
        weight_decay: float = 0.0001,
        mask: tp.Union[
            tp.Any,
            tp.Callable[
                [
                    tp.Union[
                        jnp.ndarray,
                        tp.Iterable["chex.ArrayTree"],
                        tp.Mapping[tp.Any, "chex.ArrayTree"],
                    ]
                ],
                tp.Any,
            ],
            None,
        ] = None,
    ) -> "Optimizer":
        r"""Adam with weight decay regularization.

        AdamW uses weight decay to regularise learning towards small weights, as
        this leads to better generalisation. In SGD you can also use L2 regularisation
        to implement this as an additive loss term, however L2 regularization
        does not behave as intended for adaptive gradient algorithms such as Adam.

        WARNING: Sometimes you may want to skip weight decay for BatchNorm scale or
        for the bias parameters. You can use `optax.masked` to make your own AdamW
        variant where `additive_weight_decay` is applied only to a subset of `params`.

        References:
          Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

        Args:
          learning_rate: this is a fixed global scaling factor.
          b1: the exponential decay rate to track the first moment of past gradients.
          b2: the exponential decay rate to track the second moment of past gradients.
          eps: a small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
          eps_root: (default `0`), a small constant applied to denominator inside the
            square root (as in RMSProp), to avoid dividing by zero when rescaling.
            This is needed for instance when computing (meta-)gradients through Adam.
          weight_decay: strength of the weight decay regularization.
          mask: a tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the transformation to, and `False` for those you want to skip.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(
            optax.adamw(
                learning_rate=learning_rate,
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                weight_decay=weight_decay,
                mask=mask,
            )
        )

    @staticmethod
    def adaptive_grad_clip(clipping, eps=0.001) -> "Optimizer":
        r"""Clip updates to be at most clipping * parameter_norm, unit-wise.

        References:
          [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
          Recognition Without Normalization. (https://arxiv.org/abs/2102.06171)

        Args:
          clipping: Maximum allowed ratio of update norm to parameter norm.
          eps: epsilon term to prevent clipping of zero-initialized params.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.adaptive_grad_clip(clipping=clipping, eps=eps))

    @staticmethod
    def add_decayed_weights(
        weight_decay: float = 0.0,
        mask: tp.Union[
            tp.Any,
            tp.Callable[
                [
                    tp.Union[
                        jnp.ndarray,
                        tp.Iterable["chex.ArrayTree"],
                        tp.Mapping[tp.Any, "chex.ArrayTree"],
                    ]
                ],
                tp.Any,
            ],
            None,
        ] = None,
    ) -> "Optimizer":
        r"""Add parameter scaled by `weight_decay`.

        Args:
          weight_decay: a scalar weight decay rate.
          mask: a tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the transformation to, and `False` for those you want to skip.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.add_decayed_weights(weight_decay=weight_decay, mask=mask)
        )

    @staticmethod
    def add_noise(eta: float, gamma: float, seed: int) -> "Optimizer":
        r"""Add gradient noise.

        References:
          [Neelakantan et al, 2014](https://arxiv.org/abs/1511.06807)

        Args:
          eta: base variance of the gaussian noise added to the gradient.
          gamma: decay exponent for annealing of the variance.
          seed: seed for random number generation.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.add_noise(eta=eta, gamma=gamma, seed=seed))

    @staticmethod
    def additive_weight_decay(
        weight_decay: float = 0.0,
        mask: tp.Union[
            tp.Any,
            tp.Callable[
                [
                    tp.Union[
                        jnp.ndarray,
                        tp.Iterable["chex.ArrayTree"],
                        tp.Mapping[tp.Any, "chex.ArrayTree"],
                    ]
                ],
                tp.Any,
            ],
            None,
        ] = None,
    ) -> "Optimizer":
        r"""Add parameter scaled by `weight_decay`.

        Args:
          weight_decay: a scalar weight decay rate.
          mask: a tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the transformation to, and `False` for those you want to skip.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.additive_weight_decay(weight_decay=weight_decay, mask=mask)
        )

    @staticmethod
    def apply_every(k: int = 1) -> "Optimizer":
        r"""Accumulate gradients and apply them every k steps.

        Note that if this transformation is part of a chain, the states of the other
        transformations will still be updated at every step. In particular, using
        `apply_every` with a batch size of N/2 and k=2 is not necessarily equivalent
        to not using `apply_every` with a batch size of N. If this equivalence is
        important for you, consider using the `optax.MultiSteps`.

        Args:
          k: emit non-zero gradients every k steps, otherwise accumulate them.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.apply_every(k=k))

    @staticmethod
    def apply_if_finite(
        inner: optax.GradientTransformation, max_consecutive_errors: int
    ) -> "Optimizer":
        r"""A function that wraps an optimiser to make it robust to a few NaNs or Infs.

        The purpose of this function is to prevent any optimisation to happen if the
        gradients contain NaNs or Infs. That is, when a NaN of Inf is detected in the
        gradients, the wrapped optimiser ignores that gradient update. If the NaNs or
        Infs persist after a given number of updates, the wrapped optimiser gives up
        and accepts the update.

        Args:
          inner: Inner transformation to be wrapped.
          max_consecutive_errors: Maximum number of consecutive gradient updates
            containing NaNs of Infs that the wrapped optimiser will ignore. After
            that many ignored updates, the optimiser will give up and accept.

        Returns:
          New GradientTransformation.

        """
        return Optimizer(
            optax.apply_if_finite(
                inner=inner, max_consecutive_errors=max_consecutive_errors
            )
        )

    @staticmethod
    def centralize() -> "Optimizer":
        r"""Centralize gradients.

        References:
          [Yong et al, 2020](https://arxiv.org/abs/2004.01461)

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.centralize())

    @staticmethod
    def chain(*args: optax.GradientTransformation) -> "Optimizer":
        r"""Applies a list of chainable update transformations.

        Given a sequence of chainable transforms, `chain` returns an `init_fn`
        that constructs a `state` by concatenating the states of the individual
        transforms, and returns an `update_fn` which chains the update transformations
        feeding the appropriate state to each.

        Args:
          *args: a sequence of chainable (init_fn, update_fn) tuples.

        Returns:
          A single (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.chain(*args))

    @staticmethod
    def clip(max_delta) -> "Optimizer":
        r"""Clip updates element-wise, to be between -max_delta and +max_delta.

        Args:
          max_delta: the maximum absolute value for each element in the update.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.clip(max_delta=max_delta))

    @staticmethod
    def clip_by_block_rms(threshold: float) -> "Optimizer":
        r"""Clip updates to a max rms for the gradient of each param vector or matrix.

        A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
        (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

        Args:
          threshold: the maximum rms for the gradient of each param vector or matrix.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.clip_by_block_rms(threshold=threshold))

    @staticmethod
    def clip_by_global_norm(max_norm) -> "Optimizer":
        r"""Clip updates using their global norm.

        References:
          [Pascanu et al, 2012](https://arxiv.org/abs/1211.5063)

        Args:
          max_norm: the maximum global norm for an update.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.clip_by_global_norm(max_norm=max_norm))

    @staticmethod
    def differentially_private_aggregate(
        l2_norm_clip: float, noise_multiplier: float, seed: int
    ) -> "Optimizer":
        r"""Aggregates gradients based on the DPSGD algorithm.

        WARNING: Unlike other transforms, `differentially_private_aggregate` expects
        the input updates to have a batch dimension in the 0th axis. That is, this
        function expects per-example gradients as input (which are easy to obtain in
        JAX using `jax.vmap`). It can still be composed with other transformations as
        long as it is the first in the chain.

        References:
          [Abadi et al, 2016](https://arxiv.org/abs/1607.00133)

        Args:
          l2_norm_clip: maximum L2 norm of the per-example gradients.
          noise_multiplier: ratio of standard deviation to the clipping norm.
          seed: initial seed used for the jax.random.PRNGKey

        Returns:
          A `GradientTransformation`.

        """
        return Optimizer(
            optax.differentially_private_aggregate(
                l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier, seed=seed
            )
        )

    @staticmethod
    def dpsgd(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        l2_norm_clip: float,
        noise_multiplier: float,
        seed: int,
        momentum: tp.Union[float, None] = None,
        nesterov: bool = False,
    ) -> "Optimizer":
        r"""The DPSGD optimiser.

        Differential privacy is a standard for privacy guarantees of algorithms
        learning from aggregate databases including potentially sensitive information.
        DPSGD offers protection against a strong adversary with full knowledge of the
        training mechanism and access to the model’s parameters.

        WARNING: This `GradientTransformation` expects input updates to have a batch
        dimension on the 0th axis. That is, this function expects per-example
        gradients as input (which are easy to obtain in JAX using `jax.vmap`).

        References:
          Abadi et al, 2016: https://arxiv.org/abs/1607.00133

        Args:
          learning_rate: this is a fixed global scaling factor.
          l2_norm_clip: maximum L2 norm of the per-example gradients.
          noise_multiplier: ratio of standard deviation to the clipping norm.
          seed: initial seed used for the jax.random.PRNGKey
          momentum: (default `None`), the `decay` rate used by the momentum term,
            when it is set to `None`, then momentum is not used at all.
          nesterov (default `False`): whether nesterov momentum is used.

        Returns:
          A `GradientTransformation`.

        """
        return Optimizer(
            optax.dpsgd(
                learning_rate=learning_rate,
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                seed=seed,
                momentum=momentum,
                nesterov=nesterov,
            )
        )

    @staticmethod
    def ema(
        decay: float,
        debias: bool = True,
        accumulator_dtype: tp.Union[tp.Any, None] = None,
    ) -> "Optimizer":
        r"""Compute an exponential moving average of past updates.

        Note: `trace` and `ema` have very similar but distinct updates;
        `ema = decay * ema + (1-decay) * t`, while `trace = decay * trace + t`.
        Both are frequently found in the optimisation literature.

        Args:
          decay: the decay rate for the exponential moving average.
          debias: whether to debias the transformed gradient.
          accumulator_dtype: optional `dtype` to used for the accumulator; if `None`
            then the `dtype` is inferred from `params` and `updates`.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.ema(decay=decay, debias=debias, accumulator_dtype=accumulator_dtype)
        )

    @staticmethod
    def flatten(inner: optax.GradientTransformation) -> "Optimizer":
        r"""Flattens parameters and gradients for init and update of inner transform.

        This can reduce the overhead of performing many calculations on lots of small
        variables, at the cost of slightly increased memory usage.

        Args:
          inner: Inner transformation to flatten inputs for.

        Returns:
          New GradientTransformation.

        """
        return Optimizer(optax.flatten(inner=inner))

    @staticmethod
    def fromage(learning_rate: float, min_norm: float = 1e-06) -> "Optimizer":
        r"""The Frobenius matched gradient descent (Fromage) optimiser.

        Fromage is a learning algorithm that does not require learning rate tuning.
        The optimiser is based on modelling neural network gradients via deep relative
        trust (a distance function on deep neural networks). Fromage is similar to the
        LARS optimiser and can work on a range of standard neural network benchmarks,
        such as natural language Transformers and generative adversarial networks.

        References:
          Bernstein et al, 2020: https://arxiv.org/abs/2002.03432

        Args:
          learning_rate: this is a fixed global scaling factor.
          min_norm: a minimum value that the norm of the gradient updates and the
          norm of the layer parameters can be clipped to to avoid dividing by zero
          when computing the trust ratio (as in the LARS paper).

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(optax.fromage(learning_rate=learning_rate, min_norm=min_norm))

    @staticmethod
    def identity() -> "Optimizer":
        r"""Stateless identity transformation that leaves input gradients untouched.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.identity())

    @staticmethod
    def keep_params_nonnegative() -> "Optimizer":
        r"""Modifies the updates to keep parameters non-negative, i.e. >= 0.

        This transformation ensures that parameters after the update will be
        larger than or equal to zero.
        In a chain of transformations, this should be the last one.

        WARNING: the transformation expects input params to be non-negative.
        When params is negative the transformed update will move them to 0.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.keep_params_nonnegative())

    @staticmethod
    def lamb(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-06,
        eps_root: float = 0.0,
        weight_decay: float = 0.0,
        mask: tp.Union[
            tp.Any,
            tp.Callable[
                [
                    tp.Union[
                        jnp.ndarray,
                        tp.Iterable["chex.ArrayTree"],
                        tp.Mapping[tp.Any, "chex.ArrayTree"],
                    ]
                ],
                tp.Any,
            ],
            None,
        ] = None,
    ) -> "Optimizer":
        r"""The LAMB optimiser.

        LAMB is a general purpose layer-wise adaptive large batch optimiser designed
        to provide consistent training performance across a wide range of tasks,
        including those that use attention-based models (such as Transformers) and
        ResNet-50. The optimiser is able to work with small and large batch sizes.
        LAMB was inspired by the LARS learning algorithm.

        References:
          You et al, 2019: https://arxiv.org/abs/1904.00962

        Args:
          learning_rate: this is a fixed global scaling factor.
          b1: the exponential decay rate to track the first moment of past gradients.
          b2: the exponential decay rate to track the second moment of past gradients.
          eps: a small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
          eps_root: (default `0.0`), a small constant applied to denominator inside
            the square root (as in RMSProp), to avoid dividing by zero when rescaling.
            This is needed for instance when computing (meta-)gradients through Adam.
          weight_decay (default `0.`): strength of the weight decay regularization.
          mask: a tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the transformation to, and `False` for those you want to skip.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(
            optax.lamb(
                learning_rate=learning_rate,
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                weight_decay=weight_decay,
                mask=mask,
            )
        )

    @staticmethod
    def lars(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        weight_decay: float = 0.0,
        weight_decay_mask: tp.Union[
            tp.Any,
            tp.Callable[
                [
                    tp.Union[
                        jnp.ndarray,
                        tp.Iterable["chex.ArrayTree"],
                        tp.Mapping[tp.Any, "chex.ArrayTree"],
                    ]
                ],
                tp.Any,
            ],
            None,
        ] = True,
        trust_coefficient: float = 0.001,
        eps: float = 0.0,
        trust_ratio_mask: tp.Union[
            tp.Any,
            tp.Callable[
                [
                    tp.Union[
                        jnp.ndarray,
                        tp.Iterable["chex.ArrayTree"],
                        tp.Mapping[tp.Any, "chex.ArrayTree"],
                    ]
                ],
                tp.Any,
            ],
            None,
        ] = True,
        momentum: float = 0.9,
        nesterov: bool = False,
    ) -> "Optimizer":
        r"""The LARS optimiser.

        LAMB is a layer-wise adaptive optimiser introduced to help scale SGD to
        larger batch sizes. LARS later inspired the LAMB optimiser.

        References:
          You et al, 2017: https://arxiv.org/abs/1708.03888

        Args:
          learning_rate: this is a fixed global scaling factor.
          weight_decay (default `0.`): strength of the weight decay regularization.
          weight_decay_mask: a tree with same structure as (or a prefix of) the params
            PyTree, or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the transformation to, and `False` for those you want to skip.
          trust_coefficient: a multiplier for the trust ratio.
          eps: optional additive constant in the trust ratio denominator.
          trust_ratio_mask: a tree with same structure as (or a prefix of) the params
            PyTree, or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the transformation to, and `False` for those you want to skip.
          momentum: the decay rate for momentum.
          nesterov: whether to use Nesterov momentum.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(
            optax.lars(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                weight_decay_mask=weight_decay_mask,
                trust_coefficient=trust_coefficient,
                eps=eps,
                trust_ratio_mask=trust_ratio_mask,
                momentum=momentum,
                nesterov=nesterov,
            )
        )

    @staticmethod
    def lookahead(
        fast_optimizer: optax.GradientTransformation,
        sync_period: int,
        slow_step_size: float,
        reset_state: bool = False,
    ) -> "Optimizer":
        r"""Lookahead optimizer.

        Performs steps with a fast optimizer and periodically updates a set of slow
        parameters. Optionally resets the fast optimizer state after synchronization
        by calling the init function of the fast optimizer.

        Updates returned by the lookahead optimizer should not be modified before they
        are applied, otherwise fast and slow parameters are not synchronized
        correctly.

        References:
          [Zhang et al, 2019](https://arxiv.org/pdf/1907.08610v1.pdf)

        Args:
          fast_optimizer: The optimizer to use in the inner loop of lookahead.
          sync_period: Number of fast optimizer steps to take before synchronizing
            parameters. Must be >= 1.
          slow_step_size: Step size of the slow parameter updates.
          reset_state: Whether to reset the optimizer state of the fast opimizer after
            each synchronization.

        Returns:
          A `GradientTransformation` with init and update functions. The updates
          passed to the update function should be calculated using the fast lookahead
          parameters only.

        """
        return Optimizer(
            optax.lookahead(
                fast_optimizer=fast_optimizer,
                sync_period=sync_period,
                slow_step_size=slow_step_size,
                reset_state=reset_state,
            )
        )

    @staticmethod
    def masked(
        inner: optax.GradientTransformation,
        mask: tp.Union[
            tp.Any,
            tp.Callable[
                [
                    tp.Union[
                        jnp.ndarray,
                        tp.Iterable["chex.ArrayTree"],
                        tp.Mapping[tp.Any, "chex.ArrayTree"],
                    ]
                ],
                tp.Any,
            ],
        ],
    ) -> "Optimizer":
        r"""Mask updates so only a subset of them are computed.

        For example, it is common to skip weight decay for BatchNorm scale and all
        bias parameters. In many networks, these are the only parameters with only
        one dimension. So, you may create a mask function to mask these out as
        follows::

          mask_fn = lambda p: jax.tree_map(lambda x: x.ndim != 1, p)
          weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask_fn)

        You may alternatively create the mask pytree upfront::

          mask = jax.tree_map(lambda x: x.ndim != 1, params)
          weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask)

        For the ``inner`` transform, state will only be stored for the parameters that
        have a mask value of ``True``.

        Args:
          inner: Inner transformation to mask.
          mask: a PyTree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, ``True`` for leaves/subtrees you want to
            apply the transformation to, and ``False`` for those you want to skip.

        Returns:
          New GradientTransformation wrapping ``inner``.

        """
        return Optimizer(optax.masked(inner=inner, mask=mask))

    @staticmethod
    def maybe_update(
        inner: optax.GradientTransformation,
        should_update_fn: tp.Callable[[jnp.ndarray], jnp.ndarray],
    ) -> "Optimizer":
        r"""Calls the inner update function only at certain steps.

        Creates a transformation wrapper which counts the number of times the `update`
        function has been called. This counter is passed to the `should_update_fn` to
        decide when to call the inner update function.

        When not calling the inner update function, the `updates` and the inner state
        are left untouched and just passed through. The step counter is increased
        regardless.

        Args:
          inner: the inner transformation.
          should_update_fn: this function takes in a step counter (array of shape []
            and dtype int64), and returns a boolean array of shape [].

        Returns:
          An `optax.GradientTransformation`.

        """
        return Optimizer(
            optax.maybe_update(inner=inner, should_update_fn=should_update_fn)
        )

    @staticmethod
    def multi_transform(
        transforms: tp.Mapping[tp.Hashable, optax.GradientTransformation],
        param_labels: tp.Union[tp.Any, tp.Callable[[tp.Any], tp.Any]],
    ) -> "Optimizer":
        r"""Partitions params and applies a different transformation to each subset.

        Below is an example where we apply Adam to the weights and SGD to the biases
        of a 2-layer neural network::

          import optax
          import jax
          import jax.numpy as jnp

          def map_nested_fn(fn):
            '''Recursively apply `fn` to the key-value pairs of a nested dict'''
            def map_fn(nested_dict):
              return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                      for k, v in nested_dict.items()}
            return map_fn

          params = {'linear_1': {'w': jnp.zeros((5, 6)), 'b': jnp.zeros(5)},
                    'linear_2': {'w': jnp.zeros((6, 1)), 'b': jnp.zeros(1)}}
          gradients = jax.tree_map(jnp.ones_like, params)  # dummy gradients

          label_fn = map_nested_fn(lambda k, _: k)
          tx = optax.multi_transform({'w': optax.adam(1.0), 'b': optax.sgd(1.0)},
                                     label_fn)
          state = tx.init(params)
          updates, new_state = tx.update(gradients, state, params)
          new_params = optax.apply_updates(params, updates)

        Instead of providing a ``label_fn``, you may provide a PyTree of labels
        directly.  Also, this PyTree may be a prefix of the parameters PyTree. This
        is demonstrated in the GAN pseudocode below::

          generator_params = ...
          discriminator_params = ...
          all_params = (generator_params, discriminator_params)
          param_labels = ('generator', 'discriminator')

          tx = optax.multi_transform(
              {'generator': optax.adam(0.1), 'discriminator': optax.adam(0.5)},
              param_labels)

        If you would like to not optimize some parameters, you may wrap
        ``optax.multi_transform`` with :func:`optax.masked`.

        Args:
          transforms: A mapping from labels to transformations. Each transformation
            will be only be applied to parameters with the same label.
          param_labels: A PyTree that is the same shape or a prefix of the
            parameters/updates (or a function that returns one given the parameters as
            input). The leaves of this PyTree correspond to the keys of the transforms
            (therefore the values at the leaves must be a subset of the keys).

        Returns:
          An ``optax.GradientTransformation``.

        """
        return Optimizer(
            optax.multi_transform(transforms=transforms, param_labels=param_labels)
        )

    @staticmethod
    def noisy_sgd(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        eta: float = 0.01,
        gamma: float = 0.55,
        seed: int = 0,
    ) -> "Optimizer":
        r"""A variant of SGD with added noise.

        It has been found that adding noise to the gradients can improve
        both the training error and the generalisation error in very deep networks.

        References:
          Neelakantan et al, 2014: https://arxiv.org/abs/1511.06807

        Args:
          learning_rate: this is a fixed global scaling factor.
          eta: the initial variance for the gaussian noise added to gradients.
          gamma: a parameter controlling the annealing of noise over time,
            the variance decays according to `(1+t)^-\gamma`.
          seed: the seed for the pseudo-random generation process.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(
            optax.noisy_sgd(
                learning_rate=learning_rate, eta=eta, gamma=gamma, seed=seed
            )
        )

    @staticmethod
    def radam(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-08,
        eps_root: float = 0.0,
        threshold: float = 5.0,
    ) -> "Optimizer":
        r"""The Rectified Adam optimiser.

        The adaptive learning rate in Adam has undesirably large variance in early
        stages of training, due to the limited number of training samples used to
        estimate the optimiser's statistics. Rectified Adam addresses this issue
        by analytically reducing the large variance.

        References:
          Kingma et al, 2014: https://arxiv.org/abs/1412.6980

        Args:
          learning_rate: this is a fixed global scaling factor.
          b1: the exponential decay rate to track the first moment of past gradients.
          b2: the exponential decay rate to track the second moment of past gradients.
          eps: a small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
          eps_root: (default `0`), a small constant applied to denominator inside the
            square root (as in RMSProp), to avoid dividing by zero when rescaling.
            This is needed for instance when computing (meta-)gradients through Adam.
          threshold: the threshold for variance tractability.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(
            optax.radam(
                learning_rate=learning_rate,
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                threshold=threshold,
            )
        )

    @staticmethod
    def rmsprop(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        decay: float = 0.9,
        eps: float = 1e-08,
        initial_scale: float = 0.0,
        centered: bool = False,
        momentum: tp.Union[float, None] = None,
        nesterov: bool = False,
    ) -> "Optimizer":
        r"""A flexible RMSProp optimiser.

        RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
        used for each weight is scaled by a suitable estimate of the magnitude of the
        gradients on previous steps. Several variants of RMSProp can be found
        in the literature. This alias provides an easy to configure RMSProp
        optimiser that can be used to switch between several of these variants.

        References:
          Tieleman and Hinton, 2012:
              www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
          Graves, 2013: https://arxiv.org/abs/1308.0850

        Args:
          learning_rate: this is a fixed global scaling factor.
          decay: the decay used to track the magnitude of previous gradients.
          eps: a small numerical constant to avoid dividing by zero when rescaling.
          initial_scale: (default `0.`), initialisation of accumulators tracking the
            magnitude of previous updates. PyTorch uses `0`, TF1 uses `1`. When
            reproducing results from a paper, verify the value used by the authors.
          centered: (default `False`), whether the second moment or the variance of
            the past gradients is used to rescale the latest gradients.
          momentum: (default `None`), the `decay` rate used by the momentum term,
            when it is set to `None`, then momentum is not used at all.
          nesterov (default `False`): whether nesterov momentum is used.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(
            optax.rmsprop(
                learning_rate=learning_rate,
                decay=decay,
                eps=eps,
                initial_scale=initial_scale,
                centered=centered,
                momentum=momentum,
                nesterov=nesterov,
            )
        )

    @staticmethod
    def scale(step_size: float) -> "Optimizer":
        r"""Scale updates by some fixed scalar `step_size`.

        Args:
          step_size: a scalar corresponding to a fixed scaling factor for updates.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.scale(step_size=step_size))

    @staticmethod
    def scale_by_adam(
        b1: float = 0.9, b2: float = 0.999, eps: float = 1e-08, eps_root: float = 0.0
    ) -> "Optimizer":
        r"""Rescale updates according to the Adam algorithm.

        References:
          [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

        Args:
          b1: decay rate for the exponentially weighted average of grads.
          b2: decay rate for the exponentially weighted average of squared grads.
          eps: term added to the denominator to improve numerical stability.
          eps_root: term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root))

    @staticmethod
    def scale_by_belief(
        b1: float = 0.9, b2: float = 0.999, eps: float = 0.0, eps_root: float = 1e-16
    ) -> "Optimizer":
        r"""Rescale updates according to the AdaBelief algorithm.

        References:
          [Zhuang et al, 2020](https://arxiv.org/abs/2010.07468)

        Args:
          b1: decay rate for the exponentially weighted average of grads.
          b2: decay rate for the exponentially weighted average of variance of grads.
          eps: term added to the denominator to improve numerical stability.
          eps_root: term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.scale_by_belief(b1=b1, b2=b2, eps=eps, eps_root=eps_root)
        )

    @staticmethod
    def scale_by_param_block_norm(min_scale: float = 0.001) -> "Optimizer":
        r"""Scale updates for each param block by the norm of that block's parameters.

        A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
        (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

        Args:
          min_scale: minimum scaling factor.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.scale_by_param_block_norm(min_scale=min_scale))

    @staticmethod
    def scale_by_param_block_rms(min_scale: float = 0.001) -> "Optimizer":
        r"""Scale updates by rms of the gradient for each param vector or matrix.

        A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
        (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

        Args:
          min_scale: minimum scaling factor.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.scale_by_param_block_rms(min_scale=min_scale))

    @staticmethod
    def scale_by_radam(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-08,
        eps_root: float = 0.0,
        threshold: float = 5.0,
    ) -> "Optimizer":
        r"""Rescale updates according to the Rectified Adam algorithm.

        References:
          [Liu et al, 2020](https://arxiv.org/abs/1908.03265)

        Args:
          b1: decay rate for the exponentially weighted average of grads.
          b2: decay rate for the exponentially weighted average of squared grads.
          eps: term added to the denominator to improve numerical stability.
          eps_root: term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.
          threshold: Threshold for variance tractability

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.scale_by_radam(
                b1=b1, b2=b2, eps=eps, eps_root=eps_root, threshold=threshold
            )
        )

    @staticmethod
    def scale_by_rms(
        decay: float = 0.9, eps: float = 1e-08, initial_scale: float = 0.0
    ) -> "Optimizer":
        r"""Rescale updates by the root of the exp. moving avg of the square.

        References:
          [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

        Args:
          decay: decay rate for the exponentially weighted average of squared grads.
          eps: term added to the denominator to improve numerical stability.
          initial_scale: initial value for second moment

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.scale_by_rms(decay=decay, eps=eps, initial_scale=initial_scale)
        )

    @staticmethod
    def scale_by_rss(
        initial_accumulator_value: float = 0.1, eps: float = 1e-07
    ) -> "Optimizer":
        r"""Rescale updates by the root of the sum of all squared gradients to date.

        References:
          [Duchi et al, 2011](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
          [McMahan et al., 2010](https://arxiv.org/abs/1002.4908)

        Args:
          initial_accumulator_value: Starting value for accumulators, must be >= 0.
          eps: A small floating point value to avoid zero denominator.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.scale_by_rss(
                initial_accumulator_value=initial_accumulator_value, eps=eps
            )
        )

    @staticmethod
    def scale_by_schedule(
        step_size_fn: tp.Callable[
            [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
        ]
    ) -> "Optimizer":
        r"""Scale updates using a custom schedule for the `step_size`.

        Args:
          step_size_fn: a function that takes an update count as input and proposes
            the step_size to multiply the updates by.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.scale_by_schedule(step_size_fn=step_size_fn))

    @staticmethod
    def scale_by_sm3(
        b1: float = 0.9, b2: float = 1.0, eps: float = 1e-08
    ) -> "Optimizer":
        r"""Scale updates by sm3`.

        References:
          [Anil et. al 2019](https://arxiv.org/abs/1901.11150)

        Args:
          b1: decay rate for the exponentially weighted average of grads.
          b2: decay rate for the exponentially weighted average of squared grads.
          eps: term added to the denominator to improve numerical stability.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(optax.scale_by_sm3(b1=b1, b2=b2, eps=eps))

    @staticmethod
    def scale_by_stddev(
        decay: float = 0.9, eps: float = 1e-08, initial_scale: float = 0.0
    ) -> "Optimizer":
        r"""Rescale updates by the root of the centered exp. moving average of squares.

        References:
          [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

        Args:
          decay: decay rate for the exponentially weighted average of squared grads.
          eps: term added to the denominator to improve numerical stability.
          initial_scale: initial value for second moment

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.scale_by_stddev(decay=decay, eps=eps, initial_scale=initial_scale)
        )

    @staticmethod
    def scale_by_trust_ratio(
        min_norm: float = 0.0, trust_coefficient: float = 1.0, eps: float = 0.0
    ) -> "Optimizer":
        r"""Scale updates by trust ratio`.

        References:
          [You et. al 2020](https://arxiv.org/abs/1904.00962)

        Args:
          min_norm: minimum norm for params and gradient norms; by default is zero.
          trust_coefficient: a multiplier for the trust ratio.
          eps: additive constant added to the denominator for numerical stability.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.scale_by_trust_ratio(
                min_norm=min_norm, trust_coefficient=trust_coefficient, eps=eps
            )
        )

    @staticmethod
    def scale_by_yogi(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 0.001,
        eps_root: float = 0.0,
        initial_accumulator_value: float = 1e-06,
    ) -> "Optimizer":
        r"""Rescale updates according to the Yogi algorithm.

        References:
          [Zaheer et al, 2018](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html) #pylint:disable=line-too-long

        Args:
          b1: decay rate for the exponentially weighted average of grads.
          b2: decay rate for the exponentially weighted average of variance of grads.
          eps: term added to the denominator to improve numerical stability.
          eps_root: term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.
          initial_accumulator_value: The starting value for accumulators.
            Only positive values are allowed.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.scale_by_yogi(
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                initial_accumulator_value=initial_accumulator_value,
            )
        )

    @staticmethod
    def sgd(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        momentum: tp.Union[float, None] = None,
        nesterov: bool = False,
    ) -> "Optimizer":
        r"""A canonical Stochastic Gradient Descent optimiser.

        This implements stochastic gradient descent. It also includes support for
        momentum, and nesterov acceleration, as these are standard practice when
        using stochastic gradient descent to train deep neural networks.

        References:
          Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

        Args:
          learning_rate: this is a fixed global scaling factor.
          momentum: (default `None`), the `decay` rate used by the momentum term,
            when it is set to `None`, then momentum is not used at all.
          nesterov (default `False`): whether nesterov momentum is used.

        Returns:
          A `GradientTransformation`.

        """
        return Optimizer(
            optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
        )

    @staticmethod
    def sm3(learning_rate: float, momentum: float = 0.9) -> "Optimizer":
        r"""The SM3 optimiser.

        SM3 (Square-root of Minima of Sums of Maxima of Squared-gradients Method) is a
        memory-efficient adaptive optimiser designed to decrease memory overhead when
        training very large models, such as the Transformer for machine translation,
        BERT for language modelling, and AmoebaNet-D for image classification. SM3: 1)
        applies to tensors of arbitrary dimensions and any predefined cover of the
        parameters; 2) adapts the learning rates in an adaptive and data-driven manner
        (like Adagrad and unlike Adafactor); and 3) comes with rigorous convergence
        guarantees in stochastic convex optimization settings.

        References:
          Anil et al, 2019: https://arxiv.org/abs/1901.11150

        Args:
          learning_rate: this is a fixed global scaling factor.
          momentum: the `decay` rate used by the momentum term (when it is not set to
            `None`, then momentum is not used at all).

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(optax.sm3(learning_rate=learning_rate, momentum=momentum))

    @staticmethod
    def trace(
        decay: float,
        nesterov: bool = False,
        accumulator_dtype: tp.Union[tp.Any, None] = None,
    ) -> "Optimizer":
        r"""Compute a trace of past updates.

        Note: `trace` and `ema` have very similar but distinct updates;
        `trace = decay * trace + t`, while `ema = decay * ema + (1-decay) * t`.
        Both are frequently found in the optimisation literature.

        Args:
          decay: the decay rate for the trace of past updates.
          nesterov: whether to use Nesterov momentum.
          accumulator_dtype: optional `dtype` to used for the accumulator; if `None`
            then the `dtype` is inferred from `params` and `updates`.

        Returns:
          An (init_fn, update_fn) tuple.

        """
        return Optimizer(
            optax.trace(
                decay=decay, nesterov=nesterov, accumulator_dtype=accumulator_dtype
            )
        )

    @staticmethod
    def yogi(
        learning_rate: tp.Union[
            float,
            tp.Callable[
                [tp.Union[jnp.ndarray, float, int]], tp.Union[jnp.ndarray, float, int]
            ],
        ],
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 0.001,
    ) -> "Optimizer":
        r"""The Yogi optimiser.

        Yogi is an adaptive optimiser, which provides control in tuning the effective
        learning rate to prevent it from increasing. By doing so, it focuses on
        addressing the issues of convergence and generalisation in exponential moving
        average-based adaptive methods (such as Adam and RMSprop). Yogi is a
        modification of Adam and uses the same parameters.

        References:
          Zaheer et al, 2020: http://www.sanjivk.com/yogi_nips2018.pdf

        Args:
          learning_rate: this is a fixed global scaling factor.
          b1: the exponential decay rate to track the first moment of past gradients.
          b2: the exponential decay rate to track the second moment of past gradients.
          eps: a small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.

        Returns:
          the corresponding `GradientTransformation`.

        """
        return Optimizer(optax.yogi(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps))

    @staticmethod
    def zero_nans() -> "Optimizer":
        r"""A transformation which replaces NaNs with 0.

        Zeroing values in gradients is guaranteed to produce a direction of
        non-increasing loss.

        The state of the transformation has the same tree structure as that of the
        parameters. Each leaf is a single boolean which contains True iff a NaN was
        detected in the corresponding parameter array at the last call to `update`.
        This state is not used by the transformation internally, but lets users be
        aware when NaNs have been zeroed out.

        Returns:
          A `GradientTransformation`.

        """
        return Optimizer(optax.zero_nans())

    # <<<CODEGEN END>>>
    # --------------------------------------------------------------------------


# create a decorator to copy signatures
def _copy_signatue(source_fct):
    def _copy(target_fct):
        target_fct.__signature__ = inspect.signature(source_fct)
        return target_fct

    return _copy


def create_wrapper(optax_optimizer: tp.Callable):
    name = optax_optimizer.__name__

    @_copy_signatue(optax_optimizer)
    @functools.wraps(optax_optimizer)
    def __init__(self: Optimizer, *args, **kwargs):
        self.optimizer = optax_optimizer(*args, **kwargs)

    __init__.__doc__ = optax_optimizer.__doc__

    return type(name, (Optimizer,), dict(__init__=__init__))


def _all_gradient_transformation():
    _gradient_transformation = {}
    for name, obj in inspect.getmembers(optax):
        if inspect.isfunction(obj):
            signature = inspect.signature(obj)

            # get return type
            ret_type = signature.return_annotation

            # add if return type is optax.GradientTransformation
            if ret_type == optax.GradientTransformation:
                _gradient_transformation[name] = (obj, signature)

    return _gradient_transformation


if __name__ == "__main__":
    from pathlib import Path
    import re

    gen_lines = []

    for i, (name, (obj, signature)) in enumerate(
        _all_gradient_transformation().items()
    ):
        signature: inspect.Signature
        signature_str = (
            str(signature)
            .replace("jax._src.numpy.lax_numpy", "jnp")
            .replace("typing.", "tp.")
            .replace("NoneType", "None")
            .replace("<class '", "")
            .replace("'>", "")
            .replace("ArrayTree", "chex.ArrayTree")
            .replace("optax._src.base", "optax")
        ).split(" -> ")[0]
        if "ForwardRef" in signature_str:
            signature_str = re.sub(
                r"ForwardRef\((.*?)\)", lambda m: m[1], signature_str
            )

        for type_name, type_obj in inspect.getmembers(tp):
            signature_str = re.sub(
                r"\b{type_name}\b".format(type_name=type_name),
                f"tp.{type_name}",
                signature_str,
            )

        signature_assigment_str = ", ".join(
            f"*{field}"
            if signature.parameters[field].kind == inspect.Parameter.VAR_POSITIONAL
            else f"**{field}"
            if signature.parameters[field].kind == inspect.Parameter.VAR_KEYWORD
            else f"{field}={field}"
            for field in signature.parameters
        )

        def _correct_doc(doc):
            return doc.replace("\n", "\n      ")

        gen_lines.extend(
            f'''\n
    @staticmethod
    def {name}{signature_str} -> "Optimizer":
        r"""{_correct_doc(obj.__doc__)}
        """
        return Optimizer(optax.{name}({signature_assigment_str}))
        '''.splitlines()
        )

    filepath = Path(__file__)
    lines = filepath.read_text().splitlines()

    idx_start = -1
    idx_end = -1

    for idx, line in enumerate(lines):
        if "<<<CODEGEN START>>>" in line and idx_start == -1:
            idx_start = idx
        elif "<<<CODEGEN END>>>" in line and idx_end == -1:
            idx_end = idx

    if idx_start < 0 or idx_end < 0:
        raise RuntimeError("Cannot find codegen start/end")

    new_lines = lines[: idx_start + 1] + gen_lines + lines[idx_end:]

    gen_text = "\n".join(new_lines)

    print(gen_text)
    filepath.write_text(gen_text)
