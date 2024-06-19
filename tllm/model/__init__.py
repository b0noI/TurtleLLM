import flax.linen as nn
import jax.numpy as jnp
import jax
import optax


class TurtleLlmModel(nn.Module):
  vocab_size: int
  embed_dim: int
  number_of_hidden_layers: int
  ff_dim: int
  debug: bool
  input_tokens_max_length: int

  def setup(self):
    self.tpu_model_sharding = is_tpu_avilable()
    if self.tpu_model_sharding:
      devices = jax.devices()
      mesh = jax.sharding.Mesh(devices, ('all',)) 
      self.weights_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
      self.data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("all"))
    self.prep_in_embeddings()
    self.prep_deep_layers()


  def prep_in_embeddings(self):
    self.embedding = self.param(
      'embedding',
      nn.initializers.normal(1),
      (self.vocab_size, self.embed_dim),
      jnp.float32,
    )
    self.pos_embedding = self.param(
        'pos_embedding',
        nn.initializers.normal(1),
        (1, self.input_tokens_max_length, self.embed_dim),
        jnp.float32,
    )
    if self.weights_sharding:
      self.embedding = jax.device_put(self.embedding, self.weights_sharding)
      self.pos_embedding = jax.device_put(self.pos_embedding, self.weights_sharding)
      if self.debug:
        jax.debug.visualize_array_sharding(self.embedding)
        jax.debug.visualize_array_sharding(self.pos_embedding[0])

  def prep_deep_layers(self):
    hidden_layers = []
    for i in range(self.number_of_hidden_layers):
      feed_forward = self.param(
          'ff_' + str(i),
          nn.initializers.lecun_normal(),
          (self.embed_dim, self.ff_dim),
          jnp.float32,
      )
      embed = self.param(
          'embed_' + str(i),
          nn.initializers.lecun_normal(),
          (self.ff_dim, self.embed_dim),
          jnp.float32,
      )
      if self.weights_sharding:
        feed_forward = jax.device_put(feed_forward, self.weights_sharding)
        embed = jax.device_put(embed, self.weights_sharding)
      hidden_layers.append({
        "ff": feed_forward,
        "embed": embed
      })
      self.hidden_layers = hidden_layers

  @nn.compact
  def __call__(self, input_tokens):
    if self.weights_sharding:
      # Calculate padding to make the batch size divisible by 8
      remainder = input_tokens.shape[0] % 8
      padding = (0, 8 - remainder) if remainder != 0 else (0, 0)

      # Pad input_tokens along the batch dimension
      input_tokens = jnp.pad(input_tokens, ((padding[0], padding[1]), (0, 0)))
      input_tokens = jax.device_put(input_tokens, self.data_sharding)
    x = self.embedding[input_tokens] 
    x += self.pos_embedding

    for i in range(self.number_of_hidden_layers):
      feed_forward = self.hidden_layers[i]["ff"]
      x = x @ feed_forward
      x = jax.nn.relu(x)
      embed = self.hidden_layers[i]["embed"]
      x = x @ embed
      x = jax.nn.relu(x)
      
    return x @ jnp.asarray(self.embedding).T

def is_tpu_avilable():
  devices = jax.devices()

  for device in devices:
      if device.platform == "tpu":
          return True
  else:
      return False

def calculate_loss(params, data, model):
  if is_tpu_avilable():
    remainder = data["output"].shape[0] % 8
    padding = (0, 8 - remainder) if remainder != 0 else (0, 0)

    # Pad input_tokens along the batch dimension
    data["output"] = jnp.pad(data["output"], ((padding[0], padding[1]), (0, 0)))   
  proposed_outputs = model.apply(params, data["input"]) 

  one_hot = jax.nn.one_hot(data["output"], data["vocab_size"])
  return jnp.mean(optax.softmax_cross_entropy(proposed_outputs, one_hot))
