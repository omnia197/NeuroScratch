import numpy as np

class Activation:
    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        s = self.forward(x)
        return s * (1 - s)


class Softmax(Activation):
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def derivative(self, x):
        return 1  
