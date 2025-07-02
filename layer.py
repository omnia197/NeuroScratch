import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.weights) + self.bias
        self.output = self.activation.forward(self.z)
        return self.output

    def backward(self, dA, learning_rate):
        dZ = dA * self.activation.derivative(self.z)
        dW = np.dot(self.input.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, self.weights.T)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return dA_prev
