import jax
import jax.numpy as jnp
import numpy as np

A = jnp.ones((1024, 1024))

jax.debug.visualize_array_sharding(A)

devices2d = np. reshape(jax.devices(), (4, 2))

mesh = jax.sharding.Mesh(devices2d, ["mainMesh", "mainMesh2"])
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("mainMesh", "mainMesh2"))
shardedA = jax.device_put(A, sharding)

print("======")

jax.debug.visualize_array_sharding(shardedA)