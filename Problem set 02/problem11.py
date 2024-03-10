import numpy as np

def convolution(input_array, kernel_array, stride=(1, 1)):
    input_height, input_width = input_array.shape
    kernel_height, kernel_width = kernel_array.shape
    stride_height, stride_width = stride
    # Calculate output dims
    output_height = int((input_height - kernel_height) / stride_height) + 1
    output_width = int((input_width - kernel_width) / stride_width) + 1
    # Initialize an array to store the result of the convolution
    output_array = np.zeros((output_height, output_width))
    # Iterate through the input array with the specified stride and perform the convolution
    for i in range(0, output_height * stride_height, stride_height):
        for j in range(0, output_width * stride_width, stride_width):
        # Compute the convolution at the current position
            output_array[i // stride_height, j // stride_width] = np.sum(input_array[i:i+kernel_height, j:j+kernel_width] * kernel_array)
    return output_array

def max_pooling(input_array, window_size, stride):
    input_shape = input_array.shape
    pool_size = ((input_shape[0] - window_size[0]) // stride[0]) + 1
    # Initialize output
    output_array = np.zeros((pool_size, pool_size))
    # Iterate through the input array with the specified stride and perform max pooling
    for i in range(pool_size):
        for j in range(pool_size):
            # Extract the window at the current position
            window = input_array[i * stride[0]:i * stride[0] + window_size[0], j * stride[1]:j * stride[1] + window_size[1]]
            # Find the maximum value in the window and store it in the output array
            output_array[i, j] = np.max(window)
    return output_array

input = np.array([[20, 35, 35, 35, 35, 20],
        [29, 46, 44, 42, 42, 27],
        [16, 25, 21, 19, 19, 12],
        [66, 120, 116, 154, 114, 62],
        [74, 216, 174, 252, 172, 112],
        [70, 210, 174, 252, 172, 112]])

kernel = np.array([[1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]])

stride = (1, 1)
output = convolution(input, kernel, stride)
print("Input Array:")
print(input)
print("\nKernel Array:")
print(kernel)
print("\nOutput Array after Convolution:")
print(output)

window_size = (2, 2)
stride = (2, 1)

pooling_output = max_pooling(output, window_size, stride)
print("\nMaxPooling Array:")
print(pooling_output)
