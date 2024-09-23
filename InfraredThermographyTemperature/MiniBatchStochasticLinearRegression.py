import numpy as np


class MiniBatchStochasticLinearRegression:

    def __init__(
        self, learning_rate=0.01, max_iter=1000, batch_size=32, epsilon=1e-6, bias=True
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.bias = bias
        self.weights = None

    def gradient(self, x, y):
        n_samples, n_features = x.shape
        yh = np.dot(x, self.weights)
        grad = (
            np.dot(x.T, yh - y) / n_samples
        )  # divide by N because cost is mean over N points
        return grad  # size n_features

    def gradient_descent(self, x, y):
        i = 0
        n_samples, n_features = x.shape
        grad = np.inf

        if self.weights is None:
            self.weights = np.zeros(n_features)

        while np.linalg.norm(grad) > self.epsilon and i < self.max_iter:
            index = np.random.choice(x.shape[0], self.batch_size, replace=False)
            x_batch = x[index]
            y_batch = y[index]

            grad = self.gradient(x_batch, y_batch)
            self.weights = self.weights - self.learning_rate * grad
            i += 1

    def fit(self, x, y):
        if self.bias:
            x = np.c_[np.ones(x.shape[0]), x]
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)

        self.gradient_descent(x, y)

    def predict(self, x):
        if self.bias:
            x = np.c_[np.ones(x.shape[0]), x]
        yh = np.dot(x, self.weights)
        return yh
