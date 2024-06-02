import tllm.data as tllm_data
import tllm.model as tmodel

import jax
import jax.numpy as jnp

from flax.training import train_state
import optax


if __name__ == '__main__':
    seq_per_batch_size = 5

    rngkey = jax.random.key(0)
    vocab_size = 128
    model = tmodel.TurtleLlmModel(vocab_size, 512, 4, ff_dim=2048)
    params = model.init(rngkey, jax.numpy.ones((5, seq_per_batch_size), dtype = jnp.int8))
    
    tx = optax.adam(learning_rate = 1e-2)
    state = train_state.TrainState.create(
        apply_fn = model.apply,
        params = params,
        tx = tx
    )

    texts = [
        "hello world",
        "how are you? good",
        "who are you? god",
        "where we are? home",
        "what are we doing? work",
        "what is your name? Jess",
    ]
    iter_n = 0
    while iter_n < 80:
        for input_text in texts:
            ascii_text = tllm_data.convert_text_to_ascii_numpy_array(input_text)
            batches = tllm_data.split_ascii_numpy_array_into_batches(ascii_text, seq_per_batch_size)
            input_batches = tllm_data.convert_input_batches_from_output_batches(batches)
            data = {
                "input": input_batches,
                "output": batches,
                "vocab_size": vocab_size
            }
        
            loss, grad = jax.value_and_grad(tmodel.calculate_loss)(state.params, data, model)
            state = state.apply_gradients(grads = grad)
        print(f"{iter_n} -> {loss}")
        iter_n += 1
    ascii_text = tllm_data.convert_text_to_ascii_numpy_array("how are you? good")
    batches = tllm_data.split_ascii_numpy_array_into_batches(ascii_text, seq_per_batch_size)
    input_batches = tllm_data.convert_input_batches_from_output_batches(batches)
    prediction = model.apply(params, input_batches)
    # Debugging: Print intermediate outputs
    print("Prediction shape:", prediction.shape)
    print("Prediction (raw):", prediction)
    
    predicted_indices = jnp.argmax(prediction, axis=-1).flatten()
    print("Predicted indices:", predicted_indices)
    
    predicted_text = tllm_data.convert_from_ascii_numpy_array_to_str(predicted_indices)
    print(predicted_text)
