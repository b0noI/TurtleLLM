import numpy as np


def convert_text_to_ascii_numpy_array(text):
    return np.array([ord(c) for c in text])


def split_ascii_numpy_array_into_batches(ascii_numpy_array, seq_size_per_batch):
    # Split ascii_numpy_array into batches of seq_zier_per_batch, and add 0 padding for the last batch to make shape right
    # return should be numpy array with the right shape.
    num_batches = len(ascii_numpy_array) // seq_size_per_batch
    if len(ascii_numpy_array) % seq_size_per_batch != 0:
        num_batches += 1
        # Calculate the number of batches needed
    num_batches = (len(ascii_numpy_array) + seq_size_per_batch - 1) // seq_size_per_batch
    
    # Calculate the total size of the padded array
    padded_size = num_batches * seq_size_per_batch
    
    # Create a new array of the padded size initialized with zeros
    padded_array = np.zeros(padded_size, dtype=ascii_numpy_array.dtype)
    
    # Copy the original array into the padded array
    padded_array[:len(ascii_numpy_array)] = ascii_numpy_array
    
    # Reshape the padded array into the desired batch shape
    batches = padded_array.reshape(num_batches, seq_size_per_batch)
    
    return batches


def convert_input_batches_from_output_batches(input_batches):\
    # shift right by one and add 0 as the first element per each batch
    zero_array = np.zeros((len(input_batches),  len(input_batches[0])), dtype=np.uint8)
    zero_array[:, 1:input_batches.shape[1]] = input_batches[:, 0:-1]
    return zero_array