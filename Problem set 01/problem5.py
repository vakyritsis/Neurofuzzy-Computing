import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, weights, biases, activation_functions):
        self.weights = weights
        self.biases = biases
        self.activation_functions = activation_functions
    def forward_pass(self, input_data):
        outputs = []
        out = input_data
        for i, layer in enumerate(layers):
            out = self.activation_functions[i](np.dot(self.weights[i], out) + self.biases[i])
            outputs.append(out)
        return outputs
    
def pureline(x):
    return x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def swish(x):
    return x / (1 + np.exp(-x))
weights = [
    [-2, -1],
    [[2, 1]]
]
biases = [
    [-0.5, -0.75],
    [0.5]
]

layers = [2, 1]
activation = [sigmoid, pureline]
# Create neural network instance
neural_net = NeuralNetwork(weights, biases, activation)
p_values = np.linspace(-2, 2, 100)
a11 = []
a12 = []
a2 = []
for p in p_values:
    # Perform forward pass
    layer_outputs = neural_net.forward_pass(p)
    layer_outputs_as_lists = [layer.tolist() for layer in layer_outputs]
    a11.append(layer_outputs_as_lists[0][0])
    a12.append(layer_outputs_as_lists[0][1])
    a2.append(layer_outputs_as_lists[1])
# Plot

plt.plot(p_values, a11, label='a11')
plt.plot(p_values, a12, label='a12')
plt.plot(p_values, a2, label='a2')
plt.xlabel('p')
plt.ylabel('Values')
plt.legend()
plt.show()