import numpy as np


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

        yh = np.dot(self.weights, x)
        grad = np.dot(x.T, yh - y)

        # L2 and L1 regularization, respectively.
        # Keep for report.
        # grad[1:] += self.lambdaa * self.weights[1:]
        # grad[1:] += self.lambdaa * np.sign(self.weights[1:])

        return grad  # size n_features

    def adaptive_moment_estimation(self, x_batch, y_batch):
        n_samples = x_batch.shape[0]

        for i in range(n_samples):  # Iterate over mini-batch
            rand_index = np.random.randint(n_samples)
            x_rand, y_rand = x_batch[rand_index], y_batch[rand_index]

            grad = self.gradient(x_rand, y_rand)  # Calculate gradient
            self.t += 1  # Increment time step

            # Update biased first and second moment estimates
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

            # Bias-corrected moment estimates
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)

            # Update weights using Adam's update rule
            self.weights = self.weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)

    def fit(self, x, y):
        if self.bias:
            x = np.c_[np.ones(x.shape[0]), x]

        n_samples, n_features = x.shape

        self.weights = np.zeros(n_features)
        self.m = np.zeros(n_features)  # Initialize first moment vector
        self.v = np.zeros(n_features)  # Initialize second moment vector

        for e in range(self.epoch):
            mini_batches = self.create_array_minibatch(x, y)
            for x_batch, y_batch in mini_batches:
                # self.stochastic_gradient_descent(x_batch, y_batch)
                self.adaptive_moment_estimation(x_batch, y_batch)

    def predict(self, x):
        if self.bias:
            x = np.c_[np.ones(x.shape[0]), x]
        yh = np.dot(x, self.weights)
        return yh

    def create_array_minibatch(self, x, y):
        matrix = np.c_[x, y]
        np.random.shuffle(matrix)
        mini_batches = np.array_split(matrix, self.batch_size)
        return [(batch[:, :-1], batch[:, -1]) for batch in mini_batches]
