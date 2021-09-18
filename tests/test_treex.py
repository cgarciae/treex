import typing as tp
from inspect import istraceback, signature

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import optax
import pytest

import treex as tx
from treex.tree_object import _resolve_tree_type


class Linear(tx.Module):
    w: tx.Parameter[np.ndarray]
    b: tx.Parameter[np.ndarray]
    n: tx.State[int]

    def __init__(self, din, dout, name="linear"):
        super().__init__()
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
        super().__init__()
        self.din = din
        self.dmid = dmid
        self.dout = dout
        self.name = name

        self.linear1 = Linear(din, dmid, name="linear1")
        self.linear2 = Linear(dmid, dout, name="linear2")


class TestTreex:
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

        mlp_next = mlp_params.update(mlp_states)

        assert not isinstance(mlp_next.linear1.w, tx.Nothing)
        assert not isinstance(mlp_next.linear1.b, tx.Nothing)
        assert not isinstance(mlp_next.linear1.n, tx.Nothing)

        assert not isinstance(mlp_next.linear2.w, tx.Nothing)
        assert not isinstance(mlp_next.linear2.b, tx.Nothing)
        assert not isinstance(mlp_next.linear2.n, tx.Nothing)

    def test_update_initializers(self):
        m = tx.Linear(2, 3)
        m2 = m.init(42)

        m = m.update(m2)

        assert isinstance(m.kernel, jnp.ndarray)

    def test_update_inplace(self):

        mlp = MLP(2, 3, 5)

        mlp_params = mlp.filter(tx.Parameter)
        mlp_states = mlp.filter(tx.State)

        mlp_params.update(mlp_states, inplace=True)

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

        mlp_params.update(mlp_states)

        assert not isinstance(mlp_params.linear1.w, tx.Nothing)
        assert not isinstance(mlp_params.linear1.b, tx.Nothing)
        assert isinstance(mlp_params.linear1.n, tx.Nothing)

        assert not isinstance(mlp_params.linear2.w, tx.Nothing)
        assert not isinstance(mlp_params.linear2.b, tx.Nothing)
        assert isinstance(mlp_params.linear2.n, tx.Nothing)

    def test_list(self):
        class LinearList(tx.Module):
            params: tx.Parameter[tp.List[np.ndarray]]

            def __init__(self, din, dout, name="linear"):
                super().__init__()

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
                super().__init__()
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
            def module_init(self, key):
                nonlocal n
                n = n + 1

        module = A()

        module = module.init(42)
        module = module.init(42)

        assert n == 1

    def test_initialized(self):
        class A(tx.Module):
            def module_init(self, key):
                self.x = 420

        module = A()
        assert not module.initialized

        module = module.init(42)

        assert module.x == 420
        assert module.initialized

    def test_initialized_inplace(self):
        class A(tx.Module):
            def module_init(self, key):
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

            def __init__(self, din, dmid, dout, name="mlp"):
                super().__init__()
                self.linear1 = tx.Linear(din, dmid)
                self.linear2 = tx.Linear(dmid, dout)

        mlp = MLP(2, 3, 5).init(42)

    def test_repr(self):
        class MyModule(tx.Module):
            a: tp.Dict[str, tp.List[MLP]]
            b: tx.Parameter[tp.List[tp.Union[tx.Initializer, jnp.ndarray]]]

            def __init__(self):
                super().__init__()
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
            b: tx.Parameter[tp.List[tp.Union[jnp.ndarray, tx.Initializer]]]

            def __init__(self):
                super().__init__()
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

        print(mlp.a["mlps"][1].linear2)

    def test_tabulate_inputs(self):
        class MyModule(tx.Module):
            a: tp.Dict[str, tp.List[tx.MLP]]
            b: tx.Parameter[tp.List[tp.Union[jnp.ndarray, tx.Initializer]]]

            def __init__(self):
                super().__init__()
                self.a = {"mlps": [tx.MLP([256, 1024, 512]), tx.MLP([256, 1024, 512])]}
                self.b = [
                    tx.Initializer(lambda key: jnp.zeros((512, 256))),
                    jnp.zeros((512, 128)),
                ]

            def __call__(self, x):

                y1 = self.a["mlps"][0](x)
                y2 = self.a["mlps"][1](x)

                return dict(y1=y1, y2=y2)

        mlp = MyModule().init(42)
        mlp = jax.tree_map(
            lambda x: jnp.asarray(x) if not isinstance(x, tx.Initializer) else x, mlp
        )
        # mlp = mlp.filter(tx.Parameter)

        x = np.random.uniform(size=(10, 256))

        rep = mlp.tabulate(inputs=tx.Inputs(x))

        print(rep)

    def test_resolves(self):

        # test some generic resolves to tree type
        annotation = tx.Parameter[tp.List[tp.Any]]
        tree_type = _resolve_tree_type("annotation", annotation)
        assert tree_type is tx.Parameter

        # test static generic
        annotation = tx.Static[tx.Linear]
        tree_type = _resolve_tree_type("annotation", annotation)
        assert tree_type is tx.Static

        # test static only
        annotation = tx.Static
        tree_type = _resolve_tree_type("annotation", annotation)
        assert tree_type is tx.Static

        annotation = tp.List[int]
        tree_type = _resolve_tree_type("annotation", annotation)
        assert tree_type is list

        annotation = tp.List[tx.Parameter[int]]
        tree_type = _resolve_tree_type("annotation", annotation)
        assert tree_type is tx.Parameter

        with pytest.raises(TypeError):
            annotation = tx.Parameter[tp.List[tx.State[int]]]
            _resolve_tree_type("annotation", annotation)

        with pytest.raises(TypeError):
            annotation = tx.Parameter[tp.List[tx.Linear]]
            _resolve_tree_type("annotation", annotation)

        with pytest.raises(TypeError):
            annotation = tp.Tuple[tx.Parameter, tp.List[tx.Linear]]
            _resolve_tree_type("annotation", annotation)

        annotation = tx.Static[tp.List[tx.Linear]]
        _resolve_tree_type("annotation", annotation)

    def test_static_annotation(self):
        class Mod(tx.Module):
            a: tx.Linear
            b: tx.Static[tx.Linear]

            def __init__(self):
                super().__init__()
                self.a = tx.Linear(3, 4)
                self.b = tx.Linear(3, 4)

        mod = Mod().init(42)

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
                super().__init__()
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5).init(42)

        assert "linear1" in mlp.__annotations__

    def test_auto_annotations_inserted(self):
        class MLP(tx.Module):
            def __init__(self, din, dmid, dout, name="mlp"):
                super().__init__()
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5).init(42)

        mlp.linear3 = Linear(7, 8, name="linear3").init(42)

        rep = repr(mlp)

        assert "linear3" in rep

    def test_auto_annotations_static(self):
        class MLP(tx.Module):
            linear2: tx.Static[Linear]

            def __init__(self, din, dmid, dout, name="mlp"):
                super().__init__()
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5).init(42)

        rep = repr(mlp)

        assert "linear1" in rep
        assert "linear2" not in rep

    def test_annotations_missing_field_no_error(self):
        class MLP(tx.Module):
            linear3: Linear  # missing field

            def __init__(self, din, dmid, dout, name="mlp"):
                super().__init__()
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5).init(42)

        rep = repr(mlp)

        assert "linear1" in rep
        assert "linear2" in rep

    def test_hashable(self):
        class M(tx.Module):
            a: tx.Hashable[np.ndarray]

            def __init__(self):
                super().__init__()
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
            a: tx.Parameter[np.ndarray, tx.Initializer]

            def __init__(self):
                super().__init__()
                self.a = tx.Initializer(lambda k: jax.random.uniform(k, shape=[3, 5]))

        module = MyModule()

        print(module.tabulate())

    def test_treex_filter(self):

        tree = dict(a=1, b=Linear(3, 4))

        tree2 = tx.filter(tree, tx.Parameter)
        assert isinstance(tree2["a"], tx.Nothing)

        tree2 = tx.filter(tree, lambda field: isinstance(field.value, int))
        assert tree2["a"] == 1

    def test_module_map(self):
        class A(tx.Module):
            def __init__(self):
                super().__init__()
                self.a = 1

        module = A()

        def map_fn(x):
            x.a = 2

        module2 = tx.object_apply(map_fn, module)

        assert module.a == 1
        assert module2.a == 2
