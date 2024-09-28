import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
class MiniBatchLinearRegression:
    def __init__(self, learning_rate=0.05, batch_size=32, epoch=2000, add_bias=True):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.add_bias = add_bias
        self.weights = None

    def fit(self, X, y):
        # Convert pandas series to numpy array
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # Initialize weights
        N, features = X.shape
        self.weights = np.random.randn(features + int(self.add_bias))  # +1 for bias if needed

        if self.add_bias:
            # Add bias term (column of ones) to X
            X = np.column_stack([X, np.ones(N)])
        losses = []  # To store loss over iterations
        r2s = []
        # Mini-batch gradient descent
        for epoch in range(self.epoch):
            # Shuffle the dataset
            indices = np.random.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Divide into mini-batches
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Predictions and gradient
                predictions = X_batch @ self.weights  # Linear regression hypothesis
                errors = predictions - y_batch
                gradient = X_batch.T @ errors / len(X_batch)

                # Update weights
                self.weights -= self.learning_rate * gradient
            y_pred = X @ self.weights
            mses = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            losses.append(mses)
            r2s.append(r2)
        return losses,r2s

    def predict(self, X):
        X = np.array(X)
        if self.add_bias:
            X = np.column_stack([X, np.ones(X.shape[0])])  # Add bias term
        return X @ self.weights