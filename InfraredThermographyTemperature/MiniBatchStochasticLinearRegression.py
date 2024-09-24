import numpy as np


class MiniBatchStochasticLinearRegression:

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        batch_size=32,
        epsilon=1e-6,
        epoch=100,
        bias=True,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.bias = bias
        self.epoch = epoch
        self.weights = None

    def gradient(self, x, y):

        yh = np.dot(self.weights, x)
        # n_samples should be 1 here
        grad = np.dot(x.T, yh - y)
        return grad  # size n_features

    def stochastic_gradient_descent(self, x, y):

        n_samples, n_features = x.shape
        grad = np.inf

        for i in range(n_samples):  # Iterate over all the samples in the X
            rand_index = np.random.randint(
                n_samples
            )  # fetch the index of random value in x

            x_rand, y_rand = (
                x[rand_index],
                y.iloc[rand_index],
            )  # Obtain the two corresponding sample and result

            grad = self.gradient(x_rand, y_rand)
            self.weights = self.weights - self.learning_rate * grad
            if np.linalg.norm(grad) < self.epsilon:
                break

    def fit(self, x, y):
        if self.bias:
            x = np.c_[np.ones(x.shape[0]), x]

        n_samples, n_features = x.shape

        self.weights = np.zeros(n_features)

        for e in range(self.epoch):

            self.stochastic_gradient_descent(x, y)

    def predict(self, x):
        if self.bias:
            x = np.c_[np.ones(x.shape[0]), x]
        yh = np.dot(x, self.weights)
        return yh

    def create_array_minibatch(self, x, y):
        n_samples, n_features = x.shape

        # for i in range(0, n_samples, self.batch_size)
