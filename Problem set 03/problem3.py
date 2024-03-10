import math

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
if __name__ == "__main__":
    
    # Init points and weights

    p1 = [1, 1]
    p2 = [-1, 2]
    p3 = [-2, -2]

    w1 = [0, 1]
    w2 = [1, 0]

    W = [w1, w2]

    # Learning rate 
    learning_rate = 0.5

    # One iter 
    training_set = [p1, p2, p3, p2, p3, p1]

    for p in training_set:
        W = iteration(p, W, learning_rate)

    print("Weights after trainning: ")
    print(W)


    
