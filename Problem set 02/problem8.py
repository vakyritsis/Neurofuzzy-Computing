import numpy as np
import matplotlib.pyplot as plt

# Define the function F
def F(w):
    return 0.1*w[0]**2 + 2*w[1]**2

# Define the gradient of F
def grad_F(w):
    return np.array([0.2*w[0], 4*w[1]])

def F_45(w):
    return 0.1*(w[0]+w[1])**2 + 2*(w[0]-w[1])**2

def grad_F_45(w):
    return np.array([4.2*w[0]-3.8*w[1], 4.2*w[1]-3.8*w[0]])

# Adadelta optimizer
def adadelta_optimizer(initial_w, iterations=10, rho=0.99, epsilon=1e-8, a=1):
    w = initial_w
    delta_w_sq = np.zeros_like(w)
    acc_delta_w_sq = np.zeros_like(w)
    # print(acc_delta_w_sq)
    trajectory = [w.copy()]

    for _ in range(iterations):
        gradient = grad_F_45(w)
        acc_delta_w_sq = rho * acc_delta_w_sq + (1 - rho) * gradient**2
        rms_delta_w = np.sqrt(acc_delta_w_sq + epsilon)
        delta_w = -np.sqrt(delta_w_sq + epsilon) / rms_delta_w * gradient
        w += a * delta_w
        delta_w_sq = rho * delta_w_sq + (1 - rho) * delta_w**2

        trajectory.append(w.copy())

    return np.array(trajectory)

# Plotting
def plot_contour():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = F_45([X, Y])

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.title("Contour plot of F(x)")
    plt.xlabel("w1")
    plt.ylabel("w2")

# Run Adadelta optimizer and plot trajectory
a = 3
initial_weights = np.array([3.0, 3.0])
trajectory = adadelta_optimizer(initial_weights, iterations=10000, a=a)

plot_contour()
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label='Adadelta Trajectory')
plt.legend()
plt.show()
