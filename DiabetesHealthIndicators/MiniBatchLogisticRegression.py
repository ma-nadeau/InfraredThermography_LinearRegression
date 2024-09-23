import numpy as np


def sigmoid(z):
    logistic = 1.0 / (1 + np.exp(-z))
    return logistic


class MiniBatchLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, epsilon=1e-8):
        self.weights = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
