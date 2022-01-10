import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import linear as flax_module

from treex import types
from treex.module import Module, next_key

PRNGKey = flax_module.PRNGKey
Shape = flax_module.Shape
Dtype = flax_module.Dtype
Array = flax_module.Array


class Embed(Module):
    """A linear transformation applied over the last dimension of the input.

    `Embed` is implemented as a wrapper over `flax.linen.Embed`, its constructor
    arguments accept almost the same arguments including any Flax artifacts such as initializers.
    """

    # pytree
    num_embeddings: int
    features: int
    dtype: flax_module.Dtype
    embedding_init: tp.Callable[[PRNGKey, Shape, Dtype], Array]

    embedding: tp.Optional[Array] = types.Parameter.node()

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        dtype: flax_module.Dtype = jnp.float32,
        embedding_init: tp.Callable[
            [PRNGKey, Shape, Dtype], Array
        ] = flax_module.default_embed_init,
        name: tp.Optional[str] = None,
    ):
        """
        Arguments:
            num_embeddings: number of embeddings.
            features: number of feature dimensions for each embedding.
            dtype: the dtype of the embedding vectors (default: float32).
            embedding_init: embedding initializer.
        """
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype
        self.embedding_init = embedding_init

        self.embedding = None

    @property
    def module(self) -> flax_module.Embed:
        return flax_module.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            dtype=self.dtype,
            embedding_init=self.embedding_init,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embeds the inputs along the last dimension.

        Arguments:
            inputs: input data, all dimensions are considered batch dimensions.

        Returns:
            Output which is embedded input data.  The output shape follows the input,
            with an additional `features` dimension appended.
        """
        if self.initializing():
            rngs = {"params": next_key()}
            variables = self.module.init(rngs, x)

            # Extract collections
            params = variables["params"].unfreeze()

            self.embedding = params["embedding"]

        assert self.embedding is not None
        params = {"embedding": self.embedding}

        output = self.module.apply({"params": params}, x)
        return tp.cast(jnp.ndarray, output)
