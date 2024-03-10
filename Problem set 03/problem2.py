import numpy as np
import random
import matplotlib.pyplot as plt
from math import sin, pi, exp, sqrt

# Function to approximate
def g(x):
    return 1 + sin(x * (pi) / 8)

# Activation function of 1st layer and the derivative
def radbas(n):
    return exp(-n*n)

def radbas_der(n):
    return -2*n*exp(-n*n)

# Activation function of 2st layer and the derivative
def purelin(n):
    return n

def purelin_der(n):
    return 1

def training(points, S, learning_rate, w1, b1, w2, b2, sumOfSqrError):
    for i in range(len(points)):
        n1 = []
        a1 = []
        n2 = b2
        
        for j in range(S):
            n = sqrt((points[i]-w1[j])*(points[i]-w1[j]))*b1[j]
            n1.append(n)
            a = radbas(n)
            a1.append(a)
            n2 += a * w2[j]
            
        a2 = purelin(n2)

        # Calculate error
        e = g(points[i]) - a2
        sumOfSqrError += e*e

        # Calculate sensitivities and update weights and biases
        s2 = -2*purelin_der(n2)*(e)
        s1 = []
        for j in range(S):
            s1.append(radbas_der(n1[j])*w2[j]*s2)
            w2[j] -= learning_rate*s2*a1[j]

        b2 -= learning_rate*s2

        for j in range(S):
            w1[j] -= learning_rate*s1[j]*points[i]
            b1[j] -= learning_rate*s1[j]
        

    return  w1, b1, w2, b2, sumOfSqrError

def plot_response(interval, responses, actual, S, learning_rate):
    plt.plot(interval, responses, marker='o', linestyle='-', label='Predicted')
    plt.plot(interval, actual, marker='s', linestyle='--', label='Actual')
    plt.xlabel('Points')
    plt.ylabel('Values')
    plt.title(f'Graph of Values vs. Points with S: {S} and a: {learning_rate}')
    plt.grid(True)
    plt.legend()
    plt.show()

def feedforward(point, S, w1, b1, w2, b2):
    n1 = []
    a1 = []
    n2 = b2
    
    for j in range(S):
        n = sqrt((point-w1[j])*(point-w1[j]))*b1[j]
        n1.append(n)
        a = radbas(n)
        a1.append(a)
        n2 += a * w2[j]
        
    a2 = purelin(n2)
    actual = g(point)
    return a2, actual

if __name__ == "__main__":
    # Number of random points
    num_points = 30

    # Generate 30 random points within the interval [-4, 4]
    points = [random.uniform(-4, 4) for _ in range(num_points)]
    points.sort()
    # Hyperparameters

    learning_rate = 0.1
    S = 4 # number of neurons, values are 4, 8, 12, 20

    # Weights and bias

    w1 = []
    b1 = []
    w2 = []

    for i in range(S):
        w1.append(random.uniform(-0.5, 0.5))
        b1.append(random.uniform(-0.5, 0.5))
        w2.append(random.uniform(-0.5, 0.5))
    b2 = random.uniform(-0.5, 0.5)

    sumOfSqrError = 0
    # Start training

    epochs = 6

    for epoch in range(epochs):
        w1, b1, w2, b2, sumOfSqrError = training(points, S, learning_rate, w1, b1, w2, b2, sumOfSqrError)



    print (f"Final w1: {w1}")
    print (f"Final b1: {b1}")
    print (f"Final w2: {w2}")
    print (f"Final b2: {b2}")
    print (f"Sum of squared error over the training set: {sumOfSqrError} ")

    # Feed forward for values -4 < p < 4 in order to plot the network response after training
    interval = np.linspace(-4, 4, 50)
    responses = []
    actual_values = []
    for point in interval:
        response, actual = feedforward(point, S, w1, b1, w2, b2)
        responses.append(response)
        actual_values.append(actual)

    plot_response(interval, responses, actual_values, S, learning_rate)