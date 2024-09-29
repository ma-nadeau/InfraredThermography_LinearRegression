import numpy as np
from Assignment1.Helpers import (
    adaptive_moment_estimation,
    stochastic_gradient_descent,
    create_array_minibatch,
)
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


class MiniBatchStochasticLinearRegression:

    def __init__(
        self,
        learning_rate=0.001,
        max_iter=1000,
        batch_size=32,
        epsilon=1e-6,
        epoch=100,
        bias=True,
        beta1=0.9,  # Decay rate for first moment estimate
        beta2=0.999,  # Decay rate for second moment estimate
        adam_epsilon=1e-8,  # Small constant to prevent division by zero in Ada
        lambdaa=0.01,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.bias = bias
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_epsilon = adam_epsilon
        self.epoch = epoch
        self.weights = None
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0  # Time step (iteration counter)
        self.lambdaa = lambdaa

    def gradient(self, x, y):

        yh = np.dot(x, self.weights)
        grad = np.dot(x.T, yh - y)
        grad = grad / self.batch_size
        # L2 and L1 regularization, respectively.
        # Keep for report.
        # grad[1:] += self.lambdaa * self.weights[1:]
        # grad[1:] += self.lambdaa * np.sign(self.weights[1:])

        return grad  # size n_features

    def fit(self, x, y, optimization=False):
        if self.bias:
            x = np.c_[np.ones(x.shape[0]), x]

        n_samples, n_features = x.shape
        losses = []
        r2s = []
        self.weights = np.zeros(n_features)
        self.m = np.zeros(n_features)  # Initialize first moment vector
        self.v = np.zeros(n_features)  # Initialize second moment vector

        for e in range(self.epoch):
            mini_batches = create_array_minibatch(x, y, batch_size=self.batch_size)
            for x_batch, y_batch in mini_batches:
                if optimization:
                    adaptive_moment_estimation(
                        x_batch,
                        y_batch,
                        self.weights,
                        self.m,
                        self.v,
                        self.t,
                        self.learning_rate,
                        self.beta1,
                        self.beta2,
                        self.epsilon,
                        self.gradient,
                    )
                else:
                    stochastic_gradient_descent(self, x_batch, y_batch)
            y_pred = np.dot(x, self.weights)
            mses = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            losses.append(mses)
            r2s.append(r2)
        return losses, r2s

    def predict(self, x):
        if self.bias:
            x = np.c_[np.ones(x.shape[0]), x]
        yh = np.dot(x, self.weights)
        return yh
