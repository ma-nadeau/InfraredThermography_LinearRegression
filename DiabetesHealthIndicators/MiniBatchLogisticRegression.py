from Assignment1.Helpers import *
from sklearn.metrics import log_loss

class MiniBatchLogisticRegression:
    def __init__(
            self,
            learning_rate=0.01,
            max_iter=1000,
            epsilon=1e-8,
            batch_size=10,
            epoch=100,
            add_bias=True,
            beta1=0.9,
            beta2=0.999,
    ):
        self.weights = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epoch = epoch
        self.add_bias = add_bias
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None  # (first moment)
        self.v = None  # (second moment)
        self.t = 0  # (timestep)

    def gradient(self, x, y):
        yh = sigmoid(np.dot(x, self.weights))
        grad = np.dot(x.T, yh - y)  # divide by N because cost is mean over N points
        return grad  # size n_features

    def fit(self, x, y, optimization=False):
        if self.add_bias:
            x = np.c_[np.ones(x.shape[0]), x]
        losses = []
        n_samples, n_features = x.shape

        self.weights = np.zeros(n_features)
        self.m = np.zeros(n_features)  # Initialize first moment (Added for Adam)
        self.v = np.zeros(n_features)  # Initialize second moment (Added for Adam)

        # for e in range(self.epoch):
        #     mini_batches = create_array_minibatch(x, y, batch_size=self.batch_size)
        #     for x_batch, y_batch in mini_batches:
        #         if optimization:
        #             adaptive_moment_estimation(
        #                 x_batch, y_batch, self.weights, self.m, self.v, self.t, self.learning_rate, self.beta1,
        #                 self.beta2, self.epsilon, self.gradient
        #             )
        #         else:
        #             stochastic_gradient_descent(self, x_batch, y_batch)
        #     y_train_pred = sigmoid(np.dot(x, self.weights))
        #     loss = log_loss(y, y_train_pred)
        #     losses.append(loss)
        # return losses
        # To track convergence and iterations
        iterations = 0

        for e in range(self.epoch):
            mini_batches = create_array_minibatch(x, y, batch_size=self.batch_size)
            for x_batch, y_batch in mini_batches:
                if optimization:
                    adaptive_moment_estimation(
                        x_batch, y_batch, self.weights, self.m, self.v, self.t, self.learning_rate, self.beta1,
                        self.beta2, self.epsilon, self.gradient
                    )
                else:
                    stochastic_gradient_descent(self, x_batch, y_batch)
            y_train_pred = sigmoid(np.dot(x, self.weights))
            loss = log_loss(y, y_train_pred)
            losses.append(loss)
            iterations += 1

            # Early stopping condition if loss stabilizes
            if len(losses) > 1 and abs(losses[-1] - losses[-2]) < self.epsilon:
                break

        return iterations, losses

    def predict(self, x):
        if self.add_bias:
            x = np.c_[np.ones((x.shape[0], 1)), x]
        linear_model = np.dot(x, self.weights)
        yh_real = sigmoid(linear_model)
        yh_bool = (yh_real > 0.5).astype(int)
        return yh_bool, yh_real


