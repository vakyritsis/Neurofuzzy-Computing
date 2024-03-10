import numpy as np
import random
import matplotlib.pyplot as plt


def g(x):
    return 1 + np.sin(x * (3 * np.pi) / 8)

def logsig(x):
    return 1/(1 + np.exp(-x))

def relu(x): 
    return max(x,0)

def derivative_logsig(x):
    return logsig(x) * (1 - logsig(x))

def derivative_relu(x):
    if x > 0:
        return 1
    else:
        return 0
        
def evaluate(p, W1, b1, W2, b2):
    n1 = np.dot(W1, p)+ b1
    a1 = logsig(n1)
    n2 = np.dot(a1, W2) + b2[0]
    a2 = relu(n2)
    actual = g(p)
    error = actual - a2

    return error
# init

s = 12 # number of neuros in hidden layer 2, 8, 12
max_limit = 0.5
min_limit = -0.5
a = 0.1 # learning rate

# weights and biases for first layer
W1 = [random.uniform(min_limit,max_limit) for _ in range(s)]
b1 = [random.uniform(min_limit,max_limit) for _ in range(s)]
# weights and bias for second layer
W2 = [random.uniform(min_limit,max_limit) for _ in range(s)]
b2 = [random.uniform(min_limit,max_limit)] # 1 neuron so 1 bias

# lists for plotting
error_to_plot = []
input_list = []
actual_values = []
predicted_values = []
# Initialize lists corresponding to each value of W2
w2_lists = [[] for _ in range(s)]

# Assign values to corresponding w2 lists
for i in range(s):
    w2_lists[i] = [W2[i]]

# Unpack the w2_lists into individual lists w2_1_list, w2_2_list, ..., w2_12_list
(w2_1_list, w2_2_list, w2_3_list, w2_4_list, w2_5_list, 
 w2_6_list, w2_7_list, w2_8_list, w2_9_list, w2_10_list, 
 w2_11_list, w2_12_list) = w2_lists

i = 0
while (i < 4000): 
    for p in np.arange(-2,2,0.01):

        # Feed forward for single p
        n1 = np.dot(W1, p)+ b1
        a1 = logsig(n1)
        n2 = np.dot(a1, W2) + b2[0]
        a2 = relu(n2)

        # Compute actual value
        actual = g(p)
   
        # Compute error
        error = actual - a2

        # Compute sensitivities:
        s2 = -2 * derivative_relu(n2) * error
        s1 = np.dot(derivative_logsig(n1), W2) * s2
        
        # Update weigths and biases
        W1 = np.array(W1) - a * s1 * p
        b1 = np.array(b1) - a * s1
        W2 = np.array(W2) - a * s2 * a1
        b2 = np.array(b2) - a * s2

        # Store values for plotting
        input_list.append(p)
        predicted_values.append(a2)
        actual_values.append(actual)
        error_to_plot.append(error)

        w2_1_list.append(W2[0])
        w2_2_list.append(W2[1])
        w2_3_list.append(W2[2])
        w2_4_list.append(W2[3])
        w2_5_list.append(W2[4])
        w2_6_list.append(W2[5])
        w2_7_list.append(W2[6])
        w2_8_list.append(W2[7])
        w2_9_list.append(W2[8])
        w2_10_list.append(W2[9])
        w2_11_list.append(W2[10])
        w2_12_list.append(W2[11])

        # Update i
        i = i + 1

# MAIN
print("-----EVALUATION-----")
point_not_int_dataset = 0.005
point_in_dataset = 1.5
print('error(%.3f) = ' % point_not_int_dataset, evaluate(point_not_int_dataset, W1, b1, W2, b2))
print('error(%.2f) = '% point_in_dataset, evaluate(point_in_dataset, W1, b1, W2, b2))


plt.title("Actual-Predicted values")
plt.xlabel('iter')
plt.ylabel('g(p)')
plt.plot( actual_values, 'o', label='actual values')
plt.plot( predicted_values, '+', label='Predicted values')
plt.legend()
plt.show()

plt.title("Error over iteration")
plt.xlabel('iter')
plt.ylabel('Error')
plt.plot(error_to_plot, '+', label='Error')
plt.legend()
plt.show()


plt.title("W2 Convergence over iteration")
plt.xlabel('iter')
plt.ylabel('w2 values')
plt.plot(w2_1_list, 'o', label='W2_1 Convergence')
plt.plot(w2_2_list, 'o', label='W2_2 Convergence')
plt.plot(w2_3_list, 'o', label='W2_3 Convergence')
plt.plot(w2_4_list, 'o', label='W2_4 Convergence')
plt.plot(w2_5_list, 'o', label='W2_5 Convergence')
plt.plot(w2_6_list, 'o', label='W2_6 Convergence')
plt.plot(w2_7_list, 'o', label='W2_7 Convergence')
plt.plot(w2_8_list, 'o', label='W2_8 Convergence')
plt.plot(w2_9_list, 'o', label='W2_9 Convergence')
plt.plot(w2_10_list, 'o', label='W2_10 Convergence')
plt.plot(w2_11_list, 'o', label='W2_11 Convergence')
plt.plot(w2_12_list, 'o', label='W2_12 Convergence')
plt.legend()
plt.show()