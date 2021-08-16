
import streamlit as __st
import streambook
__toc = streambook.TOCSidebar()
__toc._add(streambook.H1('Treex'))
__toc._add(streambook.H3('Initialization'))
__toc._add(streambook.H3('Modules are Pytrees'))
__toc._add(streambook.H3('Modules can be sliced'))
__toc._add(streambook.H3('Modules can be merged'))
__toc._add(streambook.H3('Modules compose'))
__toc._add(streambook.H3('Full Example'))

__toc.generate()
__st.markdown(r"""<span id='Treex'> </span>
# Treex

**Main features**:
* Modules contain their parameters
* Easy transfer learning
* Simple initialization
* No metaclass magic
* No apply method
* No need special versions of `vmap`, `jit`, and friends.

We will showcase each of the above features by creating a very contrived but complete module that will use everything from parameters, states, and random states:""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    from typing import Tuple
    import jax.numpy as jnp
    import numpy as np

    import treex as tx


    class NoisyStatefulLinear(tx.Module):
        # tree parts are defined by treex annotations
        w: tx.Parameter
        b: tx.Parameter
        count: tx.State
        rng: tx.Rng

        # other annotations are possible but ignored by type
        name: str

        def __init__(self, din, dout, name="noisy_stateful_linear"):
            self.name = name

            # Initializers only expect RNG key
            self.w = tx.Initializer(lambda k: jax.random.uniform(k, shape=(din, dout)))
            self.b = tx.Initializer(lambda k: jax.random.uniform(k, shape=(dout,)))

            # random state is JUST state, we can keep it locally
            self.rng = tx.Initializer(lambda k: k)

            # if value is known there is no need for an Initiaizer
            self.count = jnp.array(1)

        def __call__(self, x: np.ndarray) -> np.ndarray:
            assert isinstance(self.count, jnp.ndarray)
            assert isinstance(self.rng, jnp.ndarray)

            # state can easily be updated
            self.count = self.count + 1

            # random state is no different :)
            key, self.rng = jax.random.split(self.rng, 2)

            # your typical linear operation
            y = jnp.dot(x, self.w) + self.b

            # add noise for fun
            state_noise = 1.0 / self.count
            random_noise = 0.8 * jax.random.normal(key, shape=y.shape)

            return y + state_noise + random_noise

        def __repr__(self) -> str:
            return f"NoisyStatefulLinear(w={self.w}, b={self.b}, count={self.count}, rng={self.rng})"


    linear = NoisyStatefulLinear(1, 1)

    linear
__st.markdown(r"""<span id='Initialization'> </span>
### Initialization
Initialization is straightforward. The only thing you need to do is to call `init` on your module with a random key:""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    import jax

    linear = linear.init(key=jax.random.PRNGKey(42))
    linear
__st.markdown(r"""<span id='Modules are Pytrees'> </span>
### Modules are Pytrees
Modules must also be Pytrees. We can check that they are by using `tree_map` with an arbitrary function:""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    # its a pytree alright
    doubled = jax.tree_map(lambda x: 2 * x, linear)
    doubled
__st.markdown(r"""<span id='Modules can be sliced'> </span>
### Modules can be sliced
An essential feature for multiple workflows is slicing. This Module system  provides the capability of slicing based on the type of its parameters, and the `slice` method does exactly that:""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    params = linear.slice(tx.Parameter)
    states = linear.slice(tx.State)

    print(f"{params=}")
    print(f"{states=}")
__st.markdown(r"""<span id='Modules can be merged'> </span>
Notice the following:
* Both `params` and `states` are `NoisyStatefulLinear` objects, their type does not change after being sliced.
* The fields that are filtered out by the `slice` on each field get a special value of type `tx.Nothing`.

Why is this important? As we will see later, keeping parameters and state separate is helpful as they will crucially flow through different parts of `value_and_grad`.

### Modules can be merged
This is just the inverse operation to `slice`, `merge` behaves like dict's `update` but returns a new module leaving the original modules intact:""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    linear = params.merge(states)
    linear
__st.markdown(r"""<span id='Modules compose'> </span>
### Modules compose
Treex architecture easily allows you to have modules inside their modules, the same as previously. Here we will create an `MLP` class that uses two `NoisyStatefulLinear` modules: The key is to annotate the class fields.""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    class MLP(tx.Module):
        linear1: NoisyStatefulLinear
        linear2: NoisyStatefulLinear

        def __init__(self, din, dmid, dout):
            self.linear1 = NoisyStatefulLinear(din, dmid, name="linear1")
            self.linear2 = NoisyStatefulLinear(dmid, dout, name="linear2")

        def __call__(self, x: np.ndarray) -> np.ndarray:
            x = jax.nn.relu(self.linear1(x))
            x = self.linear2(x)
            return x

        def __repr__(self) -> str:
            return f"MLP(linear1={self.linear1}, linear2={self.linear2})"


    model = MLP(din=1, dmid=2, dout=1).init(key=42)
    model
__st.markdown(r"""<span id='Full Example'> </span>
### Full Example
Using the previous `model` we will show how to train it using the proposed Module system. First lets get some data:""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)


    def get_data(dataset_size: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.random.normal(size=(dataset_size, 1))
        y = 5 * x - 2 + 0.4 * np.random.normal(size=(dataset_size, 1))
        return x, y


    def get_batch(
        data: Tuple[np.ndarray, np.ndarray], batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(len(data[0]), batch_size)
        return jax.tree_map(lambda x: x[idx], data)


    data = get_data(1000)

    fig = plt.figure()  # __st
    plt.scatter(data[0], data[1])
    plt.show()
    fig  # __st
__st.markdown(r"""Now we will be reusing the previous MLP model, and we will create an optax optimizer that is used to train the model:""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    import optax

    optimizer = optax.adam(1e-2)

    params = model.slice(tx.Parameter)
    states = model.slice(tx.State)

    opt_state = optimizer.init(params)
__st.markdown(r"""Notice that we are already splitting the model into `params` and `states` since we only need to pass the `params` to the optimizer. Next, we will create the loss function, it will take the model parts and the data parts and return the loss plus the new states:""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    from functools import partial


    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params: MLP, states: MLP, x, y):

        # merge params and states to get a full model
        model: MLP = params.merge(states)

        # apply model
        pred_y = model(x)

        # MSE loss
        loss = jnp.mean((y - pred_y) ** 2)

        # new states
        states = model.slice(tx.State)

        return loss, states
__st.markdown(r"""Notice that we are merging the `params` and `states` into the complete model since we need everything in place to perform the forward pass. Also, we return the updated states from the model. The above steps are required because JAX functional API requires us to be explicit about state management.

**Note**: inside `loss_fn` (wrapped by `value_and_grad`) module can behave like a regular mutable Python object. However, every time it is treated as a pytree a new reference will be created in `jit`, `grad`, `vmap`, etc. It is essential to consider this when using functions like `vmap` inside a module, as JAX will need specific bookkeeping to manage the state correctly.

Next, we will implement the `update` function, it will look indistinguishable from your standard Haiku update, which also separates weights into `params` and `states`: """, unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    @jax.jit
    def update(params: MLP, states: MLP, opt_state, x, y):
        (loss, states), grads = loss_fn(params, states, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)

        # use regular optax
        params = optax.apply_updates(params, updates)

        return params, states, opt_state, loss
__st.markdown(r"""Finally, we create a simple training loop that performs a few thousand updates and merge `params` and `states` back into a single `model` at the end:""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    steps = 10_000

    for step in range(steps):
        x, y = get_batch(data, batch_size=32)

        params, states, opt_state, loss = update(params, states, opt_state, x, y)

        if step % 1000 == 0:
            print(f"[{step}] loss = {loss}")

    # get the final model
    model = params.merge(states)
__st.markdown(r"""Now lets generate some test data and see how our model performed:""", unsafe_allow_html=True)
with __st.echo(), streambook.st_stdout('info'):
    import matplotlib.pyplot as plt

    X_test = np.linspace(data[0].min(), data[0].max(), 100)[:, None]
    y_pred = model(X_test)

    fig = plt.figure()  # __st
    plt.scatter(data[0], data[1], label="data", color="k")
    plt.plot(X_test, y_pred, label="prediction")
    plt.legend()
    plt.show()
    fig  # __st
__st.markdown(r"""We can see that the model has learned the general trend, but because of the `NoisyStatefulLinear` modules we have a bit of noise in the predictions.""", unsafe_allow_html=True)

