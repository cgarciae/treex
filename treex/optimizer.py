import functools
import inspect
import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import treeo as to
from rich.text import Text

from treex import types, utils
from treex.treex import Treex

O = tp.TypeVar("O", bound="Optimizer")
A = tp.TypeVar("A", bound="tp.Any")


class Optimizer(Treex):
    """Wraps an optax optimizer and turn it into a Pytree while maintaining a similar API.

    The main difference with optax is that tx.Optimizer contains its own state, thus, there is
    no `opt_state`.

    Example:
    ```python
    def main():
        ...
        optimizer = tx.Optimizer(optax.adam(1e-3))
        optimizer = optimizer.init(params)
        ...

    jax.jit
    def train_step(model, x, y, optimizer):
        ...
        params = optimizer.update(grads, params)
        ...
        return model, loss, optimizer
    ```

    Notice that since the optimizer is a `Pytree` it can naturally pass through `jit`.

    ### Differences with Optax
    * `init` return a new optimizer instance, there is no `opt_state`.
    * `update` doesn't get `opt_state` as an argument, instead it performs updates
        to its internal state inplace.
    * `update` applies the updates to the params and returns them by default, use `update=False` to
        to get the param updates instead.

    Arguments:
        optimizer: An optax optimizer.
    """

    optimizer: optax.GradientTransformation
    opt_state: tp.Optional[tp.Any] = types.OptState.node(None, init=False)
    _n_params: tp.Optional[int] = to.static(None, init=False)

    # use to.field to copy class vars to instance
    _initialized: bool = to.static(False)

    def __init__(self, optimizer: optax.GradientTransformation) -> None:
        self.optimizer = optimizer

    @property
    def initialized(self) -> bool:
        return self._initialized

    def init(self: O, params: tp.Any) -> O:
        """
        Initialize the optimizer from an initial set of parameters.

        Arguments:
            params: An initial set of parameters.

        Returns:
            A new optimizer instance.
        """
        module = to.copy(self)
        params = jax.tree_leaves(params)
        module.opt_state = module.optimizer.init(params)
        module._n_params = len(params)
        module._initialized = True
        return module

    # NOTE: params are flattened because:
    # - The flat list is not a Module, thus all of its internal parameters in the list are marked as
    # OptState by a single annotation (no need to rewrite the module's annotations)
    # - It ignores the static part of Modules which if changed Optax yields an error.
    def update(
        self, grads: A, params: tp.Optional[A] = None, apply_updates: bool = True
    ) -> A:
        """
        Applies the parameters updates and updates the optimizers internal state inplace.

        Arguments:
            grads: the gradients to perform the update.
            params: the parameters to update. If `None` then `update` has to be `False`.
            apply_updates: if `False` then the updates are returned instead of being applied.

        Returns:
            The updated parameters. If `apply_updates` is `False` then the updates are returned instead.
        """
        if not self.initialized:
            raise RuntimeError("Optimizer is not initialized")

        assert self.opt_state is not None
        if apply_updates and params is None:
            raise ValueError("params must be provided if updates are being applied")

        opt_grads, treedef = jax.tree_flatten(grads)
        opt_params = jax.tree_leaves(params)

        if len(opt_params) != self._n_params:
            raise ValueError(
                f"params must have length {self._n_params}, got {len(opt_params)}"
            )
        if len(opt_grads) != self._n_params:
            raise ValueError(
                f"grads must have length {self._n_params}, got {len(opt_grads)}"
            )

        param_updates: A
        param_updates, self.opt_state = self.optimizer.update(
            opt_grads,
            self.opt_state,
            opt_params,
        )

        output: A
        if apply_updates:
            output = optax.apply_updates(opt_params, param_updates)
        else:
            output = param_updates

        return jax.tree_unflatten(treedef, output)

    # THE FOLOWING METHODS ARE AUTOMATICALLY GENERATED
    # >>> DO NOT MODIFY <<<
    # run `python -m treex.optimizer` to update methods
    # --------------------------------------------------------------------------
    # <<<CODEGEN START>>>

    # <<<CODEGEN END>>>
    # --------------------------------------------------------------------------


# create a decorator to copy signatures
def _copy_signatue(source_fct):
    def _copy(target_fct):
        target_fct.__signature__ = inspect.signature(source_fct)
        return target_fct

    return _copy


def _create_wrapper(optax_optimizer: tp.Callable):
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
    import re
    from pathlib import Path

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
