import numpy as np


def sigmoid(z):
    logistic = 1.0 / (1 + np.exp(-z))
    return logistic


class MiniBatchLogisticRegression:
    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        epsilon=1e-8,
        batch_size=10,
        epoch=100,
        add_bias=True,
    ):
        self.weights = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epoch = epoch
        self.add_bias = add_bias

    def gradient(self, x, y):
        yh = sigmoid(np.dot(x, self.weights))
        grad = np.dot(x.T, yh - y)  # divide by N because cost is mean over N points
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
                y[rand_index],
            )  # Obtain the two corresponding sample and result

            grad = self.gradient(x_rand, y_rand)
            self.weights = self.weights - self.learning_rate * grad
            if np.linalg.norm(grad) < self.epsilon:
                break

    def fit(self, x, y):
        if self.add_bias:
            x = np.c_[np.ones(x.shape[0]), x]

        n_samples, n_features = x.shape

        self.weights = np.zeros(n_features)

        for e in range(self.epoch):
            mini_batches = self.create_array_minibatch(x, y)
            for x_batch, y_batch in mini_batches:
                self.stochastic_gradient_descent(x_batch, y_batch)

    def predict(self, x):
        if self.add_bias:
            x = np.c_[np.ones((x.shape[0], 1)), x]
        linear_model = np.dot(x, self.weights)
        yh_real = sigmoid(linear_model)
        yh_bool = (yh_real > 0.5).astype(int)
        return yh_bool, yh_real

    def create_array_minibatch(self, x, y):
        matrix = np.c_[x, y]
        np.random.shuffle(matrix)
        mini_batches = np.array_split(matrix, self.batch_size)
        return [(batch[:, :-1], batch[:, -1]) for batch in mini_batches]
