import numpy as np
import math
import matplotlib.pyplot as plt

def generate_AR_samples(num_samples, seed):
    # np.random.seed(seed)
    a1, a2, a3 = 0.5, -0.1, 0.2

    samples = np.zeros(num_samples)
    samples[:3] = np.random.rand(1, 3)
    for i in range(3, num_samples):
        samples[i] = a1 * samples[i-1] + a2 * samples[i-2] + a3 * samples[i-3] + np.random.uniform(-0.25, 0.25)

    return samples

def create_dataset(samples, sequence_length):
    dataX, dataY = [], []
    for i in range(len(samples) - sequence_length):
        seq_in = samples[i:i + sequence_length]
        seq_out = samples[i + sequence_length]
        dataX.append(seq_in)
        dataY.append(seq_out)

    X = np.array(dataX)
    Y = np.array(dataY)

    return np.expand_dims(X, axis=2), np.expand_dims(Y, axis=1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTM(): 
    def __init__(self, sequence_length, hidden_dim ):
        self.learning_rate = 0.0001    
        self.nepoch = 20 
        self.T = sequence_length              # length of sequence
        self.hidden_dim = hidden_dim         
        self.output_dim = 1
        self.bptt_truncate = 5
        self.min_clip_value = -5
        self.max_clip_value = 5

        self.U = np.random.uniform(0, 1, (self.hidden_dim, self.T))
        self.W = np.random.uniform(0, 1, (self.hidden_dim, self.hidden_dim))
        self.V = np.random.uniform(0, 1, (self.output_dim, self.hidden_dim))

    def forward(self, x, prev_s):
        # forward pass
        layers = []

        for t in range(self.T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(self.U, new_input)
            self.mulw = np.dot(self.W, prev_s)
            self.add = self.mulw + mulu
            s = sigmoid(self.add)
            mulv = np.dot(self.V, s)
            layers.append({'s':s, 'prev_s':prev_s})
            prev_s = s

        return mulv, layers
        
    def backward(self, dmulv, layers, x, dU, dV, dW, dU_t, dV_t, dW_t, dU_i, dW_i):
        # backward pass
        for t in range(self.T):
            dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
            dsv = np.dot(np.transpose(self.V), dmulv)
            
            ds = dsv
            dadd = self.add * (1 - self.add) * ds
            
            dmulw = dadd * np.ones_like(self.mulw)
            dprev_s = np.dot(np.transpose(self.W), dmulw)
            for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
                ds = dsv + dprev_s
                dadd = self.add * (1 - self.add) * ds
                dmulw = dadd * np.ones_like(self.mulw)
                dmulu = dadd * np.ones_like(self.mulu)
                dW_i = np.dot(self.W, layers[t]['prev_s'])
                dprev_s = np.dot(np.transpose(self.W), dmulw)
                new_input = np.zeros(x.shape)
                new_input[t] = x[t]
                dU_i = np.dot(self.U, new_input)
                dx = np.dot(np.transpose(self.U), dmulu)
                dU_t += dU_i
                dW_t += dW_i

            dV += dV_t
            dU += dU_t
            dW += dW_t
            if dU.max() > self.max_clip_value:
                dU[dU > self.max_clip_value] = self.max_clip_value
            if dV.max() > self.max_clip_value:
                dV[dV > self.max_clip_value] = self.max_clip_value
            if dW.max() > self.max_clip_value:
                dW[dW > self.max_clip_value] = self.max_clip_value
                
            
            if dU.min() < self.min_clip_value:
                dU[dU < self.min_clip_value] = self.min_clip_value
            if dV.min() < self.min_clip_value:
                dV[dV < self.min_clip_value] = self.min_clip_value
            if dW.min() < self.min_clip_value:
                dW[dW < self.min_clip_value] = self.min_clip_value

            return dV, dU, dW
    def train(self, X, Y):
        for epoch in range(self.nepoch):
        # train model
            for i in range(Y.shape[0]):
                x, y = X[i], Y[i]
            
                prev_s = np.zeros((self.hidden_dim, 1))
                dU = np.zeros(self.U.shape)
                dV = np.zeros(self.V.shape)
                dW = np.zeros(self.W.shape)
                
                dU_t = np.zeros(self.U.shape)
                dV_t = np.zeros(self.V.shape)
                dW_t = np.zeros(self.W.shape)
                
                dU_i = np.zeros(self.U.shape)
                dW_i = np.zeros(self.W.shape)
                                
                mulv, layers = self.forward(x, prev_s)
                        # derivative of pred
                dmulv = (mulv - y)
                
                dU, dV, dW = self.backward(dmulv, layers, x, dU, dV, dW, dU_t, dV_t, dW_t, dU_i, dW_i)
                
                # update
                self.U -= self.learning_rate * dU
                self.V -= self.learning_rate * dV
                self.W -= self.learning_rate * dW
    def eval(self, ):
        preds = []
        for i in range(Y.shape[0]):
            x, y = X[i], Y[i]
            prev_s = np.zeros((hidden_dim, 1))
            # Forward pass
            for t in range(self.T):
                mulu = np.dot(self.U, x)
                mulw = np.dot(self.W, prev_s)
                add = mulw + mulu
                s = sigmoid(add)
                mulv = np.dot(self.V, s)
                prev_s = s
                
            preds.append(mulv)
            
        return np.array(preds)

if __name__ == "__main__":

    num_samples = 200
    ar_samples = generate_AR_samples(num_samples, seed=32)
    val_ar_samples = generate_AR_samples(num_samples, seed=23)

    sequence_length = 5
    hidden_dim = 100
    X, Y = create_dataset(ar_samples, sequence_length)

    X_val, Y_val = create_dataset(val_ar_samples, sequence_length)

    lstm  = LSTM(sequence_length, hidden_dim)

    lstm.train(X, Y)


    preds = lstm.eval()
    plt.plot(preds[:, 0, 0], 'g', label="predict")
    plt.plot(Y[:, 0], 'r', label="actual")
    plt.legend()

    plt.show()


    # preds = []
    # for i in range(Y_val.shape[0]):
    #     x, y = X_val[i], Y_val[i]
    #     prev_s = np.zeros((hidden_dim, 1))
    #     # Forward pass
    #     for t in range(T):
    #         mulu = np.dot(U, x)
    #         mulw = np.dot(W, prev_s)
    #         add = mulw + mulu
    #         s = sigmoid(add)
    #         mulv = np.dot(V, s)
    #         prev_s = s
            
    #     preds.append(mulv)
        
    # preds = np.array(preds)
    # mse = np.mean((preds-Y_val)**2)
    # print(mse)
    # plt.plot(preds[:, 0, 0], 'g')
    # plt.plot(Y_val[:, 0], 'r')
    # plt.legend()

    # plt.show()


