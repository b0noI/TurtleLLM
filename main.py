import tllm.data as data
import tllm.model as tmodel

import jax
import jax.numpy as jnp

from flax.training import train_state
import optax


if __name__ == '__main__':
    seq_per_batch_size = 5
    input_text = "Hello world"
    ascii_text = data.convert_text_to_ascii_numpy_array(input_text)
    print("ASCII text: " + str(ascii_text))
    batches = data.split_ascii_numpy_array_into_batches(ascii_text, seq_per_batch_size)
    print("Batches (output):")
    for batch in batches:
        print(batch)
    input_batches = data.convert_input_batches_from_output_batches(batches)
    print("Batches (input):")
    print(input_batches)
    print("----")

    rngkey = jax.random.key(0)
    vocab_size = 128
    model = tmodel.TurtleLlmModel(vocab_size, 512, 4, ff_dim=2048)
    params = model.init(rngkey, jax.numpy.ones((len(batches), seq_per_batch_size), dtype = jnp.int8))
    
    tx = optax.adam(learning_rate = 1e-3)
    state = train_state.TrainState.create(
        apply_fn = model.apply,
        params = params,
        tx = tx
    )

    # propsoed_out = model.apply(params, input_batches)
    # print("Output: ")
    # print("Shape: " + str(propsoed_out.shape))
    # print(str(propsoed_out))
    data = {
        "input": input_batches,
        "output": batches,
        "vocab_size": vocab_size
    }
    loss = tmodel.calcualte_loss(params, data, model)
    print("loss: " + str(loss))