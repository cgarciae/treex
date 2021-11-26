import typing as tp
from dataclasses import dataclass
from inspect import istraceback, signature

import cloudpickle
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import optax
import pytest

import treex as tx


class Linear(tx.Module):
    w: np.ndarray = tx.Parameter.node()
    b: np.ndarray = tx.Parameter.node()
    n: int = tx.State.node()

    def __init__(self, din, dout, name="linear"):

        self.din = din
        self.dout = dout
        self.w = np.random.uniform(size=(din, dout))
        self.b = np.random.uniform(size=(dout,))
        self.n = 1
        self.name = name


class MLP(tx.Module):
    linear1: Linear
    linear2: Linear

    def __init__(self, din, dmid, dout, name="mlp"):

        self.din = din
        self.dmid = dmid
        self.dout = dout
        self.name = name

        self.linear1 = Linear(din, dmid, name="linear1")
        self.linear2 = Linear(dmid, dout, name="linear2")


def _get_all_vars(cls):
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__dict__"):
            d.update(vars(c))
    return d


class TestTreex:
    def test_vars_inheritance(self):
        class A:
            a = 1

        class B(A):
            b = 2

        v = _get_all_vars(B)

        v

    def test_flatten_nothing(self):
        x = [(1, 2), (3, tx.Nothing())]
        assert jax.tree_leaves(x) == [1, 2, 3]

        flat_with_nothing = jax.tree_flatten(x, lambda x: isinstance(x, tx.Nothing))[0]

        assert flat_with_nothing == [1, 2, 3, tx.Nothing()]

    def test_flatten(self):

        mlp = MLP(2, 3, 5)

        flat = jax.tree_leaves(mlp)

        assert len(flat) == 6

    def test_flatten_slice(self):

        mlp = MLP(2, 3, 5).filter(tx.State)

        flat = jax.tree_leaves(mlp)

        assert len(flat) == 2

    def test_flatten_slice_merging(self):

        mlp = MLP(2, 3, 5).filter(tx.State)

        flat = jax.tree_flatten(mlp, lambda x: isinstance(x, tx.Nothing))[0]

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
        mlp_params = mlp.filter(tx.Parameter)

        assert not isinstance(mlp_params.linear1.w, tx.Nothing)
        assert not isinstance(mlp_params.linear1.b, tx.Nothing)
        assert isinstance(mlp_params.linear1.n, tx.Nothing)

        assert not isinstance(mlp_params.linear2.w, tx.Nothing)
        assert not isinstance(mlp_params.linear2.b, tx.Nothing)
        assert isinstance(mlp_params.linear2.n, tx.Nothing)

        # states
        mlp_states = mlp.filter(tx.State)

        assert isinstance(mlp_states.linear1.w, tx.Nothing)
        assert isinstance(mlp_states.linear1.b, tx.Nothing)
        assert not isinstance(mlp_states.linear1.n, tx.Nothing)

        assert isinstance(mlp_states.linear2.w, tx.Nothing)
        assert isinstance(mlp_states.linear2.b, tx.Nothing)
        assert not isinstance(mlp_states.linear2.n, tx.Nothing)

    def test_update(self):

        mlp = MLP(2, 3, 5)

        mlp_params = mlp.filter(tx.Parameter)
        mlp_states = mlp.filter(tx.State)

        mlp_next = mlp_params.merge(mlp_states)

        assert not isinstance(mlp_next.linear1.w, tx.Nothing)
        assert not isinstance(mlp_next.linear1.b, tx.Nothing)
        assert not isinstance(mlp_next.linear1.n, tx.Nothing)

        assert not isinstance(mlp_next.linear2.w, tx.Nothing)
        assert not isinstance(mlp_next.linear2.b, tx.Nothing)
        assert not isinstance(mlp_next.linear2.n, tx.Nothing)

    def test_update_initializers(self):
        x = np.random.uniform(size=(5, 2))
        m = tx.Linear(3)
        m2 = m.init(42, x)

        m = m.merge(m2)

        assert isinstance(m.kernel, jnp.ndarray)

    def test_update_inplace(self):

        mlp = MLP(2, 3, 5)

        mlp_params = mlp.filter(tx.Parameter)
        mlp_states = mlp.filter(tx.State)

        mlp_params.merge(mlp_states, inplace=True)

        assert not isinstance(mlp_params.linear1.w, tx.Nothing)
        assert not isinstance(mlp_params.linear1.b, tx.Nothing)
        assert not isinstance(mlp_params.linear1.n, tx.Nothing)

        assert not isinstance(mlp_params.linear2.w, tx.Nothing)
        assert not isinstance(mlp_params.linear2.b, tx.Nothing)
        assert not isinstance(mlp_params.linear2.n, tx.Nothing)

    def test_update_not_inplace(self):

        mlp = MLP(2, 3, 5)

        mlp_params = mlp.filter(tx.Parameter)
        mlp_states = mlp.filter(tx.State)

        mlp_params.merge(mlp_states)

        assert not isinstance(mlp_params.linear1.w, tx.Nothing)
        assert not isinstance(mlp_params.linear1.b, tx.Nothing)
        assert isinstance(mlp_params.linear1.n, tx.Nothing)

        assert not isinstance(mlp_params.linear2.w, tx.Nothing)
        assert not isinstance(mlp_params.linear2.b, tx.Nothing)
        assert isinstance(mlp_params.linear2.n, tx.Nothing)

    def test_list(self):
        class LinearList(tx.Module):
            params: tp.List[np.ndarray] = tx.Parameter.node()

            def __init__(self, din, dout, name="linear"):

                self.din = din
                self.dout = dout
                self.params = [
                    np.random.uniform(size=(din, dout)),
                    np.random.uniform(size=(dout,)),
                ]
                self.name = name

        linear = LinearList(2, 3, name="mlp")

        @jax.jit
        def idfn(x):
            return x

        assert not isinstance(linear.params[0], jnp.DeviceArray)
        assert not isinstance(linear.params[1], jnp.DeviceArray)

        linear = idfn(linear)

        assert isinstance(linear.params[0], jnp.DeviceArray)
        assert isinstance(linear.params[1], jnp.DeviceArray)

    def test_treelist(self):
        class MLP(tx.Module):
            linears: tp.List[Linear]

            def __init__(self, din, dmid, dout, name="mlp"):

                self.linears = [
                    Linear(din, dmid, name="linear1"),
                    Linear(dmid, dout, name="linear2"),
                ]

        mlp = MLP(2, 3, 5)

        @jax.jit
        def idfn(x):
            return x

        assert not isinstance(mlp.linears[0].w, jnp.DeviceArray)
        assert not isinstance(mlp.linears[0].b, jnp.DeviceArray)
        assert not isinstance(mlp.linears[0].n, jnp.DeviceArray)

        assert not isinstance(mlp.linears[1].w, jnp.DeviceArray)
        assert not isinstance(mlp.linears[1].b, jnp.DeviceArray)
        assert not isinstance(mlp.linears[1].n, jnp.DeviceArray)

        mlp = idfn(mlp)

        assert isinstance(mlp.linears[0].w, jnp.DeviceArray)
        assert isinstance(mlp.linears[0].b, jnp.DeviceArray)
        assert isinstance(mlp.linears[0].n, jnp.DeviceArray)

        assert isinstance(mlp.linears[1].w, jnp.DeviceArray)
        assert isinstance(mlp.linears[1].b, jnp.DeviceArray)
        assert isinstance(mlp.linears[1].n, jnp.DeviceArray)

    def test_idenpotent_init(self):
        n = 0

        class A(tx.Module):
            def rng_init(self):
                nonlocal n
                n = n + 1

        module = A()

        module = module.init(42)
        module = module.init(42)

        assert n == 1

    def test_initialized(self):
        class A(tx.Module):
            def rng_init(self):
                self.x = 420

        module = A()
        assert not module.initialized

        module = module.init(42)

        assert module.x == 420
        assert module.initialized

    def test_initialized_inplace(self):
        class A(tx.Module):
            def rng_init(self):
                self.x = 420

        module = A()
        assert not module.initialized

        module.init(42, inplace=True)

        assert module.x == 420
        assert module.initialized

    def test_train(self):

        mlp = MLP(2, 3, 5).init(42)

        assert mlp.training
        assert mlp.linear1.training
        assert mlp.linear2.training

        mlp = mlp.eval()

        assert not mlp.training
        assert not mlp.linear1.training
        assert not mlp.linear2.training

        mlp = mlp.train()

        assert mlp.training
        assert mlp.linear1.training
        assert mlp.linear2.training

    def test_train_inplace(self):

        mlp = MLP(2, 3, 5).init(42)

        assert mlp.training
        assert mlp.linear1.training
        assert mlp.linear2.training

        mlp.eval(inplace=True)

        assert not mlp.training
        assert not mlp.linear1.training
        assert not mlp.linear2.training

        mlp.train(inplace=True)

        assert mlp.training
        assert mlp.linear1.training
        assert mlp.linear2.training

    def test_multiple_initializers(self):
        class MLP(tx.Module):
            linear1: tx.Linear
            linear2: tx.Linear

            def __init__(self, dmid, dout, name="mlp"):

                self.linear1 = tx.Linear(dmid)
                self.linear2 = tx.Linear(dout)

            def __call__(self, x):
                return self.linear2(self.linear1(x))

        x = np.random.uniform(size=(2, 3))
        mlp = MLP(2, 3, 5).init(42, x)

    def test_repr(self):
        class MyModule(tx.Module):
            a: tp.Dict[str, tp.List[MLP]]
            b: tp.List[tp.Union[tx.Initializer, jnp.ndarray]] = tx.Parameter.node()

            def __init__(self):

                self.a = {"mlps": [MLP(2, 3, 5), MLP(2, 3, 5)]}
                self.b = [
                    tx.Initializer(lambda key: jnp.zeros((10, 4))),
                    jnp.zeros((5, 13)),
                ]

        mlp = MyModule()  # .init(42)
        mlp = jax.tree_map(
            lambda x: jnp.asarray(x) if not isinstance(x, tx.Initializer) else x, mlp
        )
        mlp = mlp.filter(tx.Parameter)

        rep = repr(mlp)

        rep

    def test_tabulate(self):
        class MyModule(tx.Module):
            a: tp.Dict[str, tp.List[MLP]]
            b: tp.List[tp.Union[jnp.ndarray, tx.Initializer]] = tx.Parameter.node()

            def __init__(self):

                self.a = {"mlps": [MLP(256, 1024, 512), MLP(256, 1024, 512)]}
                self.b = [
                    tx.Initializer(lambda key: jnp.zeros((512, 256))),
                    jnp.zeros((512, 128)),
                ]

        mlp = MyModule()  # .init(42)
        mlp = jax.tree_map(
            lambda x: jnp.asarray(x) if not isinstance(x, tx.Initializer) else x, mlp
        )
        # mlp = mlp.filter(tx.Parameter)

        rep = mlp.tabulate()

        print(rep)

        assert '.a["mlps"]' in rep
        assert "b:" in rep

        print(mlp.a["mlps"][1].linear2)

    def test_tabulate_inputs(self):
        class MyModule(tx.Module):
            a: tp.Dict[str, tp.List[tx.MLP]]
            b: tp.List[tp.Union[jnp.ndarray]] = tx.Parameter.node()

            def __init__(self):

                self.a = {"mlps": [tx.MLP([2, 3]), tx.MLP([2, 3])]}
                self.b = [
                    jnp.zeros((512, 256)),
                    jnp.zeros((512, 128)),
                ]

            def __call__(self, x):

                y1 = self.a["mlps"][0](x)
                y2 = self.a["mlps"][1](x)

                return dict(y1=y1, y2=y2)

        x = np.random.uniform(size=(5, 1))
        mlp = MyModule().init(42, x)
        mlp = jax.tree_map(
            lambda x: jnp.asarray(x) if not isinstance(x, tx.Initializer) else x, mlp
        )
        # mlp = mlp.filter(tx.Parameter)

        x = np.random.uniform(size=(5, 1))

        rep = mlp.tabulate(inputs=tx.Inputs(x))

        print(rep)

        assert "(\x1b[32m5, 1\x1b[0m)" in rep
        assert "y1:" in rep
        assert "y2:" in rep

    def test_static_annotation(self):
        class Mod(tx.Module):
            a: tx.Linear
            b: tx.Linear = tx.static()

            def __init__(self):

                self.a = tx.Linear(4)
                self.b = tx.Linear(4)

            def __call__(self, x):
                x = self.a(x)
                return x

        x = np.random.uniform(size=(5, 4))
        mod = Mod().init(42, x)

        assert len(jax.tree_leaves(mod)) == 2

        assert mod.a.initialized
        assert mod.a.kernel is not None
        assert mod.a.bias is not None

        assert not mod.b.initialized
        assert mod.b.kernel is None
        assert mod.b.bias is None

    def test_auto_annotations(self):
        class MLP(tx.Module):
            def __init__(self, din, dmid, dout, name="mlp"):

                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5).init(42)

        assert "linear1" in mlp.field_metadata

    def test_auto_annotations_inserted(self):
        class MLP(tx.Module):
            def __init__(self, din, dmid, dout, name="mlp"):

                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5).init(42)

        mlp.linear3 = Linear(7, 8, name="linear3").init(42)

        mlp.check_metadata_updates()  # find field

        assert "linear3" in mlp.field_metadata

    def test_auto_annotations_static(self):
        class MLP(tx.Module):
            linear2: Linear = tx.static()

            def __init__(self, din, dmid, dout, name="mlp"):

                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5).init(42)

        assert "linear1" in mlp.field_metadata
        assert not mlp.field_metadata["linear2"].node

    def test_annotations_missing_field_no_error(self):
        class MLP(tx.Module):
            linear3: Linear  # missing field

            def __init__(self, din, dmid, dout, name="mlp"):

                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5).init(42)

        assert "linear1" in mlp.field_metadata
        assert "linear2" in mlp.field_metadata

    def test_hashable(self):
        class M(tx.Module):
            a: tx.Hashable[np.ndarray]

            def __init__(self):

                self.a = tx.Hashable(np.ones((3, 4), dtype=np.float32))

        m = M().init(42)

        N = 0

        @jax.jit
        def f(x):
            nonlocal N
            N += 1
            return x

        m = f(m)
        assert N == 1

        m = f(m)
        assert N == 1

        m.a = tx.Hashable(np.zeros((3, 4), dtype=np.float32))

        m = f(m)
        assert N == 2

        m = f(m)
        assert N == 2

    def test_initializer(self):
        init = tx.Initializer(lambda k: jax.random.uniform(k, shape=[3, 5]))

        @jax.jit
        def f(x):
            return x

        init2 = f(init)

    def test_uninitialized_tabulate(self):
        class MyModule(tx.Module):
            a: tp.Union[np.ndarray, tx.Initializer] = tx.Parameter.node()

            def __init__(self):

                self.a = tx.Initializer(lambda k: jax.random.uniform(k, shape=[3, 5]))

        module = MyModule()

        print(module.tabulate())

    def test_treex_filter(self):
        tree = dict(a=1, b=Linear(3, 4))

        tree2 = tx.filter(tree, tx.Parameter)
        assert isinstance(tree2["a"], tx.Nothing)

        tree2 = tx.filter(tree, lambda field: isinstance(field.value, int))
        assert tree2["a"] == 1

    def test_treex_parameter_filter(self):
        mlp = MLP(2, 3, 5)
        mlp.linear1.freeze(inplace=True)

        params = mlp.trainable_parameters()
        assert isinstance(params.linear1.b, tx.Nothing)
        assert isinstance(params.linear1.w, tx.Nothing)
        assert not isinstance(params.linear2.b, tx.Nothing)
        assert not isinstance(params.linear2.w, tx.Nothing)

    def test_module_map(self):
        class A(tx.Module):
            def __init__(self):

                self.a = 1

        module = A()

        def map_fn(x):
            x.a = 2

        module2 = tx.apply(map_fn, module)

        assert module.a == 1
        assert module2.a == 2

    def test_dataclass(self):
        @dataclass
        class M(tx.Module):
            pass

        module = M()

        assert module._initialized is False

    def test_cloudpickle(self):
        cloudpickle.dumps(tx.Linear(10))


@dataclass
class LazyLinear(tx.Module):
    features: int
    w: tp.Optional[jnp.ndarray] = tx.Parameter.node(None)
    b: tp.Optional[jnp.ndarray] = tx.Parameter.node(None)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.initializing():
            self.w = jax.random.uniform(
                tx.next_key(), shape=[x.shape[-1], self.features]
            )
            self.b = jnp.zeros([self.features])

        assert self.w is not None and self.b is not None

        return jnp.dot(x, self.w) + self.b


class TestCompact:
    def test_shape_inference(self):

        x = jnp.ones((5, 2))
        module = LazyLinear(1).init(42, x)

        y = module(x)

        assert y.shape == (5, 1)

    def test_init_error(self):

        x = jnp.ones((5, 2))
        module = LazyLinear(1)

        with pytest.raises(RuntimeError):
            y = module(x)

    def test_compact(self):
        @dataclass
        class MLP(tx.Module):
            dmid: int
            dout: int

            @tx.compact
            def __call__(self, x):
                x = LazyLinear(self.dmid)(x)
                x = jax.nn.relu(x)
                x = LazyLinear(self.dout)(x)
                return x

        x = jnp.ones((5, 2))
        mlp = MLP(3, 4).init(42, x)

        y = mlp(x)

        assert y.shape == (5, 4)
        assert "lazy_linear" in vars(mlp)
        assert "lazy_linear2" in vars(mlp)

    def test_compact_module(self):
        @tx.compact_module
        def MLP(x, dmid: int, dout: int) -> jnp.ndarray:
            x = LazyLinear(dmid)(x)
            x = jax.nn.relu(x)
            x = LazyLinear(dout)(x)
            return x

        x = jnp.ones((5, 2))
        mlp = MLP().init(42, (x, 3, 4))

        y = mlp(x, 3, 4)

        assert y.shape == (5, 4)
        assert "lazy_linear" in vars(mlp)
        assert "lazy_linear2" in vars(mlp)

    def test_nested_compact_module(self):
        @dataclass(repr=False)
        class MLP(tx.Module):
            dmid: int
            dout: int

            @tx.compact
            def __call__(self, x):
                @tx.compact_module
                def Block(x, dmid: int, dout: int) -> jnp.ndarray:
                    x = LazyLinear(dmid)(x)
                    x = jax.nn.relu(x)
                    x = LazyLinear(dout)(x)
                    return x

                x = Block()(x, self.dmid, self.dout)
                x = Block()(x, self.dmid, self.dout)
                return x

        x = jnp.ones((5, 2))
        mlp = MLP(3, 4).init(42, x)

        y = mlp(x)
        y = mlp(x)

        assert y.shape == (5, 4)
        assert "block" in vars(mlp)
        assert "block2" in vars(mlp)
