import flax.linen as nn
import jax.numpy as jnp
import jax
import optax


class TurtleLlmModel(nn.Module):
  vocab_size: int
  embed_dim: int
  layers: int
  ff_dim: int

  @nn.compact
  def __call__(self, input_tokens):
    '''
        input_tokens is [BATCH, SEQUENCE]
    '''
    print(input_tokens.shape)
    embedding = self.param(
        'embedding',
        nn.initializers.normal(1),
        (self.vocab_size, self.embed_dim),
        jnp.float32,
    )

    x = jnp.asarray(embedding)[input_tokens] # BATCH, SEQUENCE, EMBED
    print(str(x))
    pos_embedding = self.param(
        'pos_embedding',
        nn.initializers.normal(1),
        (1, list(input_tokens.shape)[1], self.embed_dim),
        jnp.float32,
    )
    x += jnp.asarray(pos_embedding)
    print(str(x.shape))
    for i in range(self.layers):
      feed_forward = self.param(
          'ff_' + str(i),
          nn.initializers.lecun_normal(),
          (self.embed_dim, self.ff_dim),
          jnp.float32,
      )
      x = x @ jnp.asarray(feed_forward)
      x = jax.nn.relu(x)
      embed = self.param(
          'embed_' + str(i),
          nn.initializers.lecun_normal(),
          (self.ff_dim, self.embed_dim),
          jnp.float32,
      )
      x = x @ jnp.asarray(embed)
      x = jax.nn.relu(x)
      
    return x @ jnp.asarray(embedding).T


def calcualte_loss(params, data, model):
  proposed_outputs = model.apply(params, data["input"])
  one_hot = jax.nn.one_hot(data["output"], vocab_dim)
  return jnp.mean(optax.softmax_cross_entropy(proposed_outputs, one_hot))
