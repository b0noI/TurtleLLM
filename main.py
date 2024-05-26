import tllm.data as data

if __name__ == '__main__':
    input_text = "Hello world"
    ascii_text = data.convert_text_to_ascii_numpy_array(input_text)
    print(ascii_text)
    batches = data.split_ascii_numpy_array_into_batches(ascii_text, 5)
    for batch in batches:
        print(batch)
    input_batches = data.convert_input_batches_from_output_batches(batches)
    print(input_batches)