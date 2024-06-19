import jax
import jax.numpy as jnp
import numpy as np

A = jnp.ones((1024, 1024))
B = jnp.ones((1024, 1024))

jax.debug.visualize_array_sharding(A)

print(jax.devices())

devices2d = np. reshape(jax.devices(), (4, 2))

mesh = jax.sharding.Mesh(devices2d, ["mainMesh", "mainMesh2"])
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("mainMesh", "mainMesh2"))
shardedA = jax.device_put(A, sharding)
shardedB = jax.device_put(B, sharding)

print("===A===")

jax.debug.visualize_array_sharding(shardedA)

print("===B===")

jax.debug.visualize_array_sharding(shardedB)

print("===C===")

jax.debug.visualize_array_sharding(A + B)