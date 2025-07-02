import numpy as np
from layer import Layer
import pickle

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.losses = []  

    def add(self, input_size, output_size, activation):
        self.layers.append(Layer(input_size, output_size, activation))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y, learning_rate):
        output = self.forward(X)
        loss = -np.mean(np.sum(y * np.log(output + 1e-9), axis=1))
        self.losses.append(loss)
        dA = (output - y) / y.shape[0]

        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)

        return loss


    def train(self, X, y, epochs, batch_size, learning_rate, X_val=None, y_val=None):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                x_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.backward(x_batch, y_batch, learning_rate)

            print(f"Epoch {epoch}, Loss: {self.losses[-1]:.4f}")
            if X_val is not None and y_val is not None:
                acc = self.evaluate(X_val, y_val)
                print(f"Val Accuracy: {acc:.2f}")

    def evaluate(self, X, y):
        out = self.forward(X)
        preds = np.argmax(out, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(preds == labels)
    

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.layers, f)

    def load(self, path="model.pkl"):
        with open(path, "rb") as f:
            self.layers = pickle.load(f)
