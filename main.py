from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
from activation import ReLU, Softmax
from neuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

def to_numpy(dataset):
    X, y = [], []
    for img, label in dataset:
        X.append(img.numpy().reshape(-1))  
        one_hot = np.zeros(10)
        one_hot[label] = 1
        y.append(one_hot)
    return np.array(X), np.array(y)

def load_mnist():
    train_set = MNIST(root='data', train=True, download=True, transform=ToTensor())
    test_set = MNIST(root='data', train=False, download=True, transform=ToTensor())
    X_train, y_train = to_numpy(train_set)
    X_test, y_test = to_numpy(test_set)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_mnist()

model = NeuralNetwork()
model.add(784, 128, ReLU())
model.add(128, 10, Softmax())

model.train(X_train, y_train, epochs=10, batch_size=64, learning_rate=0.1,
            X_val=X_test, y_val=y_test)
model.save("mnist_model.pkl")
plt.plot(model.losses)
plt.title("Training Loss per Batch")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.show()
