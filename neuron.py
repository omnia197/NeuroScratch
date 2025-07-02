import numpy as np


class Neuron:
    def __init__(self):
        self.weight = np.random.rand()
        self.last_input = None

    def update_weight(self, delta, lr):
        self.weight = self.weight - lr * delta * self.last_input

    def get_weight(self):
        return self.weight
    
    def activate(self, f):
        self.last_input = f
        return self.weight * f
        