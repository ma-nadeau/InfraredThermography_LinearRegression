import numpy as np


def sigmoid(z):
    logistic = 1.0 / (1 + np.exp(-z))
    return logistic


class MiniBatchLogisticRegression:
    def __init__(
        self, learning_rate=0.01, max_iter=1000, epsilon=1e-8, batch_size=10, epoch=100
    ):
        self.weights = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epoch = epoch

    def gradient(self, x, y):
        n_samples, n_features = x.shape
        yh = sigmoid(np.dot(x, self.weights))
        grad = (
            np.dot(x.T, yh - y) / n_samples
        )  # divide by N because cost is mean over N points
        return grad  # size n_features
