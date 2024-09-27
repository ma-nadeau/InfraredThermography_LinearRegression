import numpy as np


class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, optimization=False):
        """From the matrix X and vector Y, fit computes the weights for the linear regression

        Args:
            X (Matrix): A matrix containing the features
            Y (Vector): A vector containing the target (aveOralM)
            optimization (boolean): A boolean indicating if optimization should be performed using ADAM.
        """
        # Add a column of 1 to X -> [1, x1, x2, ..., x_D]^T
        x_w0 = np.c_[np.ones(X.shape[0]), X]  # c_ -> concatenates along second axis
        self.weights = np.linalg.pinv(x_w0.T @ x_w0) @ x_w0.T @ Y

    def predict(self, X):
        """
        Once a model has been trained, we predict y_hat using the weights and the matrix X
        Args:
            X (Matrix): A matrix containing the features

        Returns:
            y_hat (vector): A vector containing the predicted targets
        """
        # Add a column of 1 to X -> [1, x1, x2, ..., x_D]^T
        x_w0 = np.c_[np.ones(X.shape[0]), X]  # c_ -> concatenates along second axis
        y_hat = x_w0 @ self.weights  # y_hat = X * w
        return y_hat  # Returns the prediction made by our model
