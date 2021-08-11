from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util
import numpy as np
import optax

import treex as tx


Parameter = tx.annotation("Parameter", np.ndarray)
State = tx.annotation("State", np.ndarray)


class Linear(tx.Treex):
    w: Parameter
    b: Parameter
    count: State

    def __init__(self, din, dout, key, name="linear"):

        k1, k2 = jax.random.split(key, 2)
        self.din = din
        self.dout = dout

        self.w = jax.random.uniform(k1, shape=(din, dout))
        self.b = jax.random.uniform(k2, shape=(dout,))
        self.count = jnp.array(0)

        self.name = name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.count = self.count + 1
        return jnp.dot(x, self.w) + self.b


class MLP(tx.Treex):
    linear1: Linear
    linear2: Linear

    def __init__(self, din, dmid, dout, key, name="linear"):

        k1, k2 = jax.random.split(key, 2)
        self.linear1 = Linear(din, dmid, k1, name="linear1")
        self.linear2 = Linear(dmid, dout, k2, name="linear2")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Toy data
def get_data(dataset_size, *, key):
    x = jrandom.normal(key, (dataset_size, 1))
    y = 5 * x - 2
    return x, y


# Simple dataloader
def dataloader(arrays, key, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    dataset_size=10000,
    batch_size=256,
    learning_rate=3e-3,
    steps=400,
    width_size=8,
    seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    data = get_data(dataset_size, key=data_key)
    data = dataloader(data, batch_size=batch_size, key=loader_key)

    model = MLP(din=1, dmid=width_size, dout=1, key=model_key)

    @partial(jax.value_and_grad, has_aux=True)
    def loss(params: MLP, states: MLP, x, y):
        # merge params and states to get a full model
        model = params.merge(states)

        # apply model
        pred_y = model(x)

        # get output states
        states = model.slice(State)

        return jnp.mean((y - pred_y) ** 2), states

    @jax.jit
    def update_fn(params: MLP, states: MLP, opt_state, x, y):
        (value, states), grads = loss(params, states, x, y)
        updates, opt_state = optim.update(grads, opt_state, params)

        # use regular optax
        params = optax.apply_updates(params, updates)

        return params, states, opt_state, value

    # split model into parameters and states
    params = model.slice(Parameter)
    states = model.slice(State)

    optim = optax.sgd(learning_rate)
    opt_state = optim.init(params)

    for step, (x, y) in zip(range(steps), data):
        params, states, opt_state, value = update_fn(params, states, opt_state, x, y)
        print(step, value)

    assert isinstance(params, MLP)
    assert isinstance(states, MLP)

    # get the final model
    model = params.merge(states)

    print(f"\n{model.linear1.count=}, {model.linear2.count=}")


if __name__ == "__main__":
    main()
