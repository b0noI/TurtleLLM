import jax
import jax.numpy as jnp
import datetime

MATRIX_DIM = int(32768 / 2 / 2)
STEPS = 10

A = jnp.ones( (MATRIX_DIM, MATRIX_DIM), dtype = jnp.float32 )
B = jnp.ones( (MATRIX_DIM, MATRIX_DIM), dtype = jnp.float32 )

num_bites = A.size * 4 # since float32
# total_num_bytes_crossing_memory = 3 * num_bites
total_flops = MATRIX_DIM * MATRIX_DIM

def time_it(f, *args):
  times = []
  jax.block_until_ready(f(*args))
  for _ in range(STEPS):
    start_time = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    end_time = datetime.datetime.now()
    times.append((end_time - start_time).total_seconds())
  average_time = sum(times) / len(times)

  print(f"time: {average_time:.6f}")
  
@jax.jit
def sum_matrix(a, b):
    return jax.nn.relu(a + b)

time_it(sum_matrix, A, B)

# jit_sum = jax.jit(sum_matrix)

# time_it(jit_sum, A, B)