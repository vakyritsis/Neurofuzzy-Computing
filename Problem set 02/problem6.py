import numpy as np

# Initial conditions of network

w1 = 1
w2 = -1
w12 = 0.5
b = 1
p1 = 0
p2 = 1
t = 0.75
# learning rate 
learning_rate = 1

# Feed forward phase 

n = w1 * p1 + w2 * p2 + w12 * p1 * p2 + b
a = np.tanh(n)
# Compute the error

error = (t - a)**2

# Update the parameters (weights and bias) using the learning rule we have presented

w1 = w1 + 2 * learning_rate * (t - a) * (1 - np.tanh(a)**2) * p1
w2 = w2 + 2 * learning_rate * (t - a) * (1 - np.tanh(a)**2) * p2
w12 = w12 + 2 * learning_rate * (t - a) * (1 - np.tanh(a)**2) *p1 * p2
b = b + 2 * learning_rate * (t - a) * (1 - np.tanh(a)**2)

print('w1 new = ',w1)
print('w2 new = ',w2)
print('w1,2 new = ',w12)
print('b new = ',b)