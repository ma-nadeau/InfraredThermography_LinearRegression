import numpy as np


def sigmoid(z):
    logistic = 1.0 / (1 + np.exp(-z))
    return logistic


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, epsilon=1e-8, add_bias=True):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.add_bias = add_bias

    def gradient(self, x, y):
        n_samples, n_features = x.shape
        yh = sigmoid(np.dot(x, self.weights))
        grad = (
            np.dot(x.T, yh - y) / n_samples
        )  # divide by N because cost is mean over N points
        return grad  # size n_features

    def fit(self, x, y):
        if self.add_bias:
            x = np.c_[np.ones((x.shape[0], 1)), x]  # Adding columns of 1 to X
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        grad = np.inf
        i = 0
        while np.linalg.norm(grad) > self.epsilon and i < self.max_iter:
            grad = self.gradient(x, y)
            self.weights = self.weights - self.learning_rate * grad
            i += 1

    def predict(self, x):
        if self.add_bias:
            x = np.c_[np.ones((x.shape[0], 1)), x]
        linear_model = np.dot(x, self.weights)
        yh_real = sigmoid(linear_model)
        yh_bool = (yh_real > 0.5).astype(int)
        return yh_bool, yh_real
