import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from treex import __version__
from treex.base import Treex, Nothing, Parameter, State


@jax.tree_util.register_pytree_node_class
class Linear(Treex):
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


@jax.tree_util.register_pytree_node_class
class MLP(Treex):
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

        assert not isinstance(mlp_params.linear1.w, Nothing)
        assert not isinstance(mlp_params.linear1.b, Nothing)
        assert isinstance(mlp_params.linear1.n, Nothing)

        assert not isinstance(mlp_params.linear2.w, Nothing)
        assert not isinstance(mlp_params.linear2.b, Nothing)
        assert isinstance(mlp_params.linear2.n, Nothing)

        # states
        mlp_states = mlp.slice(State)

        assert isinstance(mlp_states.linear1.w, Nothing)
        assert isinstance(mlp_states.linear1.b, Nothing)
        assert not isinstance(mlp_states.linear1.n, Nothing)

        assert isinstance(mlp_states.linear2.w, Nothing)
        assert isinstance(mlp_states.linear2.b, Nothing)
        assert not isinstance(mlp_states.linear2.n, Nothing)

    def test_merge(self):

        mlp = MLP(2, 3, 5)

        mlp_params = mlp.slice(Parameter)
        mlp_states = mlp.slice(State)

        mlp_next = mlp_params.merge(mlp_states)

        assert not isinstance(mlp_next.linear1.w, Nothing)
        assert not isinstance(mlp_next.linear1.b, Nothing)
        assert not isinstance(mlp_next.linear1.n, Nothing)

        assert not isinstance(mlp_next.linear2.w, Nothing)
        assert not isinstance(mlp_next.linear2.b, Nothing)
        assert not isinstance(mlp_next.linear2.n, Nothing)
