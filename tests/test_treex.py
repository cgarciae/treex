from typing import Union
from train_mlp import Parameter
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import treex as tx


Parameter = tx.annotation("Parameter", np.ndarray)
State = tx.annotation("State", Union[np.ndarray, int])


class Linear(tx.Treex):
    w: Parameter
    b: Parameter
    n: State

    def __init__(self, din, dout, name="linear"):

        self.din = din
        self.dout = dout
        self.w = np.random.uniform(size=(din, dout))
        self.b = np.random.uniform(size=(dout,))
        self.n = 1
        self.name = name


class MLP(tx.Treex):
    linear1: Linear
    linear2: Linear

    def __init__(self, din, dmid, dout, name="mlp"):
        self.linear1 = Linear(din, dmid, name="linear1")
        self.linear2 = Linear(dmid, dout, name="linear2")


class TestTreex:
    def test_flatten(self):

        mlp = MLP(2, 3, 5)

        flat = jax.tree_leaves(mlp)

        assert len(flat) == 6

    def test_is_tree(self):

        mlp = MLP(2, 3, 5)

        @jax.jit
        def idfn(x):
            return x

        assert not isinstance(mlp.linear1.w, jnp.DeviceArray)
        assert not isinstance(mlp.linear1.b, jnp.DeviceArray)
        assert not isinstance(mlp.linear1.n, jnp.DeviceArray)

        assert not isinstance(mlp.linear2.w, jnp.DeviceArray)
        assert not isinstance(mlp.linear2.b, jnp.DeviceArray)
        assert not isinstance(mlp.linear1.n, jnp.DeviceArray)

        mlp = idfn(mlp)

        assert isinstance(mlp.linear1.w, jnp.DeviceArray)
        assert isinstance(mlp.linear1.b, jnp.DeviceArray)
        assert isinstance(mlp.linear1.n, jnp.DeviceArray)

        assert isinstance(mlp.linear2.w, jnp.DeviceArray)
        assert isinstance(mlp.linear2.b, jnp.DeviceArray)
        assert isinstance(mlp.linear2.n, jnp.DeviceArray)

    def test_filter(self):

        mlp = MLP(2, 3, 5)

        # params
        mlp_params = mlp.slice(Parameter)

        assert mlp_params.linear1.w is not None
        assert mlp_params.linear1.b is not None
        assert mlp_params.linear1.n is None

        assert mlp_params.linear2.w is not None
        assert mlp_params.linear2.b is not None
        assert mlp_params.linear2.n is None

        # states
        mlp_states = mlp.slice(State)

        assert mlp_states.linear1.w is None
        assert mlp_states.linear1.b is None
        assert mlp_states.linear1.n is not None

        assert mlp_states.linear2.w is None
        assert mlp_states.linear2.b is None
        assert mlp_states.linear2.n is not None

    def test_merge(self):

        mlp = MLP(2, 3, 5)

        mlp_params = mlp.slice(Parameter)
        mlp_states = mlp.slice(State)

        mlp_next = mlp_params.merge(mlp_states)

        assert mlp_next.linear1.w is not None
        assert mlp_next.linear1.b is not None
        assert mlp_next.linear1.n is not None

        assert mlp_next.linear2.w is not None
        assert mlp_next.linear2.b is not None
        assert mlp_next.linear2.n is not None
