import jax
import jax.numpy as jnp
import datetime

MATRIX_DIM = 32768
STEPS = 10

A = jnp.ones( (MATRIX_DIM, MATRIX_DIM), dtype = jnp.float32 )
B = jnp.ones( (MATRIX_DIM, MATRIX_DIM), dtype = jnp.float32 )

num_bites = A.size * 4 # since float32
# total_num_bytes_crossing_memory = 3 * num_bites
total_flops = MATRIX_DIM * MATRIX_DIM

jax.profiler.start_trace("/tmp/trace1")
start_time = datetime.datetime.now()
for _ in range(STEPS):
    C = A + B
end_time = datetime.datetime.now()
#jax.profiler.stop_trace()

average_time = (end_time - start_time).total_seconds() / STEPS

print(f"time: {average_time}, tfps: {total_flops/average_time / 10**12}")
