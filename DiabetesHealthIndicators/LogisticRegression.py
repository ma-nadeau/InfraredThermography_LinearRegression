import numpy as np
from Assignment1.Helpers import sigmoid


class LogisticRegression:
    """
    Logistic Regression class that implements logistic regression using gradient descent.

    Attributes:
    -----------
    learning_rate : float
        The learning rate for gradient descent (default is 0.01).
    max_iter : int
        The maximum number of iterations for each epoch (default is 1000).
    epsilon : float
        Convergence threshold for gradient descent (default is 1e-8).
    add_bias : bool
        Whether or not to add a bias term to the input features (default is True).
    epochs : int
        Number of epochs for the training process (default is 10).
    weights : numpy.ndarray
        Weights of the logistic regression model.
    bias : numpy.ndarray or None
        Bias term (if added) of the logistic regression model.

    Methods:
    --------
    gradient(x, y):
        Computes the gradient of the loss function with respect to the weights.
    fit(x, y, optimization=False):
        Trains the model using gradient descent.
    predict(x):
        Predicts binary class labels and probabilities for the input data.
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, epsilon=1e-8, add_bias=True, epochs=10):
        """
        Initializes the LogisticRegression model with the given hyperparameters.

        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent.
        max_iter : int
            Maximum iterations for each epoch.
        epsilon : float
            Convergence threshold for gradient descent.
        add_bias : bool
            If True, add bias to the model by augmenting features with a bias column.
        epochs : int
            Number of epochs for the training process.
        """
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.add_bias = add_bias
        self.epochs = epochs

    def gradient(self, x, y):
        """
        Computes the gradient of the cost function with respect to the weights.

        Parameters:
        -----------
        x : numpy.ndarray
            Input data (features), shape (n_samples, n_features).
        y : numpy.ndarray
            Target values (binary labels), shape (n_samples,).

        Returns:
        --------
        grad : numpy.ndarray
            The gradient vector with respect to weights, shape (n_features,).
        """
        n_samples, n_features = x.shape
        yh = sigmoid(np.dot(x, self.weights))  # Predicted probabilities using sigmoid
        grad = np.dot(x.T, (yh - y)) / n_samples  # Compute gradient of the cost function
        return grad

    def fit(self, x, y, optimization=False):
        """
        Trains the logistic regression model using gradient descent.

        Parameters:
        -----------
        x : numpy.ndarray
            Training data (features), shape (n_samples, n_features).
        y : numpy.ndarray
            Target values (binary labels), shape (n_samples,).
        optimization : bool, optional
            If set to True, applies optimization techniques (not implemented here).

        Returns:
        --------
        None
        """
        if self.add_bias:
            x = np.c_[np.ones((x.shape[0], 1)), x]  # Add bias term by adding a column of ones

        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)  # Initialize weights with zeros
        grad = np.inf  # Initial gradient set to infinity for while loop

        # Iterate through epochs
        for e in range(self.epochs):
            i = 0
            # Continue until the gradient is smaller than epsilon or max_iter is reached
            while np.linalg.norm(grad) > self.epsilon and i < self.max_iter:
                grad = self.gradient(x, y)  # Compute the gradient
                self.weights -= self.learning_rate * grad  # Update weights
                i += 1  # Increment the iteration count

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
            x = np.c_[np.ones((x.shape[0], 1)), x]  # Add bias term to test data

        linear_model = np.dot(x, self.weights)  # Compute linear model output
        yh_real = sigmoid(linear_model)  # Apply sigmoid to get probabilities
        yh_bool = (yh_real > 0.5).astype(int)  # Convert probabilities to binary labels (0 or 1)

        return yh_bool, yh_real
