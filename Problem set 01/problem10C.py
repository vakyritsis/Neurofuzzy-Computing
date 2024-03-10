import numpy as np
import matplotlib.pyplot as plt
# Patterns for class A
class_a = [(0, 0), (0, 1), (1, 0), (-1, -1)]
# Patterns for class B
class_b = [(2.1, 0), (0, -2.5), (1.6, -1.6)]
# Unpack x and y coordinates for plotting
x_a, y_a = zip(*class_a)
x_b, y_b = zip(*class_b)

# Plotting decision boundary (a line)
w0 = 0.5989
w1 = -0.5556
b = -0.7187
x_decision = np.linspace(-3, 3, 100) # X-axis range
y_decision = (-w0 * x_decision - b) / w1 # Decision boundary equation
plt.plot(x_decision, y_decision, label='Decision Boundary', color='black')
# Plotting class A and class B points
plt.scatter(x_a, y_a, label='Class A')
plt.scatter(x_b, y_b, label='Class B')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Patterns for Classes A and B')
plt.grid(True)
plt.show()