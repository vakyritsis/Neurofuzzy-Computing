import math
import matplotlib.pyplot as plt
import numpy as np
def competitive_layer(input_list):
    """
    Competitive layer that returns the index of the biggest number,
    with the smallest index breaking ties.
    """
    max_index = 0
    max_value = input_list[0]

    for i in range(1, len(input_list)):
        if input_list[i] > max_value:
            max_index = i
            max_value = input_list[i]
        elif input_list[i] == max_value:
            max_index = min(max_index, i)

    result = [0] * len(input_list)
    result[max_index] = 1

    return result

# Kohonen Rule for update
def update_weights(a1, W, learning_rate, p):
    index = a1.index(1)

    difference = [p[i] - W[index][i] for i in range(len(p))]
    product = [learning_rate * element for element in difference]
    W[index] = [W[index][i] + product[i] for i in range(len(product))]

    return W

def iteration(p, W, learning_rate):
    n1 = []
    for i in range(0, len(W)):
        n1.append(-math.sqrt((W[i][0] - p[0])**2 + (W[i][1] - p[1])**2))

    a1  = competitive_layer(n1)
    W = update_weights(a1, W, learning_rate, p)

    return W

def plot(p1, p2, p3, W):

    # Extract x and y coordinates
    p1_x, p1_y = p1
    p2_x, p2_y = p2
    p3_x, p3_y = p3
    w1_x, w1_y = W[0]
    w2_x, w2_y = W[1]

    vector1 = np.array(W[0])
    vector2 = np.array(W[1])

    # Calculate the angles of the vectors in radians
    angle_radians_vector1 = np.arctan2(vector1[1], vector1[0])
    angle_radians_vector2 = np.arctan2(vector2[1], vector2[0])

    # Calculate the angle difference between the two vectors
    angle_difference = (angle_radians_vector2 - angle_radians_vector1) /2 + angle_radians_vector1

    # Plot the line with the angle difference from the origin (0,0)
    x_line = np.linspace(-1, 1, 100)
    y_line = np.tan(angle_difference) * x_line
    plt.plot(x_line, y_line, label=f'Line between vectors', linestyle='--', color='c')

    # Plot points
    plt.plot(p1_x, p1_y, 'ro', label='p1')
    plt.plot(p2_x, p2_y, 'go', label='p2')
    plt.plot(p3_x, p3_y, 'bo', label='p3')
   
    plt.quiver(0, 0, w1_x, w1_y, angles='xy', scale_units='xy', scale=1, color='m', label='w1')
    plt.quiver(0, 0, w2_x, w2_y, angles='xy', scale_units='xy', scale=1, color='y', label='w2')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of Points with Vectors')
    plt.legend()

    # Set axis limits
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)

    # Show plot
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    
    # Init points and weights

    p1 = [2, 0]
    p2 = [0, 1]
    p3 = [2, 2]

    w1 = [1, 0]
    w2 = [-1, 0]

    W = [w1, w2]

    # Learning rate 
    learning_rate = 0.5
 
    training_set1 = [p1, p2, p3, p2, p3, p1]
    training_set2 = [p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1,p1, p2, p3, p2, p3, p1]
    print(W)
    plot(p1, p2, p3, W)

    # In order to make the full training swape training_set_1 with training_set_2 and remove the plot()-print() from the for loop in order to escape plotting every updated weights
    # Just put the plot()-print() outside the for loop to show the final state.
    for p in training_set1:
        W = iteration(p, W, learning_rate)
        # print(W)
        # plot(p1, p2, p3, W)
    

    print("Trained weights are: ")
    print(W)
    plot(p1, p2, p3, W)


    
