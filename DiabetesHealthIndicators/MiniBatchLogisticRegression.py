from Assignment1.Helpers import *
from sklearn.metrics import log_loss


class MiniBatchLogisticRegression:
    """
    Mini-batch Logistic Regression with options for Adam optimization.

    Attributes:
    -----------
    learning_rate : float
        Learning rate for the optimization algorithm (default is 0.01).
    max_iter : int
        Maximum number of iterations for each mini-batch (default is 1000).
    epsilon : float
        Convergence threshold for gradient updates (default is 1e-8).
    batch_size : int
        Number of samples in each mini-batch (default is 10).
    epoch : int
        Number of complete passes over the training dataset (default is 100).
    add_bias : bool
        Whether to add a bias term to the input features (default is True).
    beta1 : float
        Decay rate for the first moment estimate in Adam optimization (default is 0.9).
    beta2 : float
        Decay rate for the second moment estimate in Adam optimization (default is 0.999).
    m : numpy.ndarray
        First moment vector for Adam optimization.
    v : numpy.ndarray
        Second moment vector for Adam optimization.
    t : int
        Time step for Adam optimization, keeps track of iterations.

    Methods:
    --------
    gradient(x, y):
        Computes the gradient of the cost function with respect to the weights.
    fit(x, y, optimization=False):
        Trains the model using mini-batch gradient descent or Adam optimizer.
    predict(x):
        Predicts binary class labels and probabilities for the input data.
    """

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
        """
        Initializes the MiniBatchLogisticRegression model with the specified parameters.

        Parameters:
        -----------
        learning_rate : float
            Learning rate for the gradient descent or Adam optimizer.
        max_iter : int
            Maximum number of iterations for each mini-batch.
        epsilon : float
            Convergence threshold for the optimization process.
        batch_size : int
            The number of samples in each mini-batch for gradient descent.
        epoch : int
            The number of full passes through the dataset.
        add_bias : bool
            If True, a bias term is added to the input features.
        beta1 : float
            Decay rate for the first moment estimate in Adam.
        beta2 : float
            Decay rate for the second moment estimate in Adam.
        """
        self.weights = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epoch = epoch
        self.add_bias = add_bias
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None  # First moment vector for Adam optimization
        self.v = None  # Second moment vector for Adam optimization
        self.t = 0  # Time step counter for Adam

    def gradient(self, x, y):
        """
        Computes the gradient of the log-loss function with respect to the weights.

        Parameters:
        -----------
        x : numpy.ndarray
            Input data (features), shape (n_samples, n_features).
        y : numpy.ndarray
            Target values (binary labels), shape (n_samples,).

        Returns:
        --------
        grad : numpy.ndarray
            The gradient vector of the log-loss function with respect to weights, shape (n_features,).
        """
        yh = sigmoid(np.dot(x, self.weights))  # Predicted probabilities using the sigmoid function
        grad = np.dot(x.T, (yh - y)) / self.batch_size  # Compute gradient
        return grad

    def fit(self, x, y, optimization=False):
        """
        Trains the logistic regression model using mini-batch gradient descent or Adam optimizer.

        Parameters:
        -----------
        x : numpy.ndarray
            Training data (features), shape (n_samples, n_features).
        y : numpy.ndarray
            Target values (binary labels), shape (n_samples,).
        optimization : bool, optional
            If set to True, the Adam optimizer is used. Otherwise, simple gradient descent is used.

        Returns:
        --------
        iterations : int
            Number of iterations run before convergence.
        losses : list
            List of log-loss values computed at each epoch.
        """
        if self.add_bias:
            # Add a column of ones to the input data for the bias term
            x = np.c_[np.ones(x.shape[0]), x]

        losses = []  # To store loss values for each epoch
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)  # Initialize weights to zero
        self.m = np.zeros(n_features)  # Initialize first moment vector (for Adam)
        self.v = np.zeros(n_features)  # Initialize second moment vector (for Adam)

        iterations = 0  # Track the number of iterations

        # Iterate over epochs
        for e in range(self.epoch):
            # Create mini-batches from the dataset
            mini_batches = create_array_minibatch(x, y, batch_size=self.batch_size)
            for x_batch, y_batch in mini_batches:
                if optimization:
                    # Use Adam optimization
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
                    # Use simple stochastic gradient descent
                    stochastic_gradient_descent(self, x_batch, y_batch)

            # Predict on the full dataset after each epoch
            y_train_pred = sigmoid(np.dot(x, self.weights))
            loss = log_loss(y, y_train_pred)  # Compute log-loss
            losses.append(loss)  # Store the loss

            iterations += 1

            # Early stopping condition: break if loss stabilizes
            if len(losses) > 1 and abs(losses[-1] - losses[-2]) < self.epsilon:
                break

        return iterations, losses

    def predict(self, x):
        """
        Predicts binary class labels and probabilities for the input data.

        Parameters:
        -----------
        x : numpy.ndarray
            Test data (features), shape (n_samples, n_features).

        Returns:
        --------
        yh_bool : numpy.ndarray
            Predicted binary class labels (0 or 1), shape (n_samples,).
        yh_real : numpy.ndarray
            Predicted probabilities for class 1, shape (n_samples,).
        """
        if self.add_bias:
            # Add a bias term (column of ones) to the input data
            x = np.c_[np.ones((x.shape[0], 1)), x]

        # Compute the linear model
        linear_model = np.dot(x, self.weights)
        yh_real = sigmoid(linear_model)  # Apply the sigmoid function to get probabilities
        yh_bool = (yh_real > 0.5).astype(int)  # Convert probabilities to binary labels (0 or 1)

        return yh_bool, yh_real
