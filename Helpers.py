import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)


# Obtain correlation matrix.
def get_correlation(df, target):
    correlation_matrix = df.corr()
    correlation = correlation_matrix[target]
    return correlation


# Obtain Variance Inflation Factor (VIF)
def calculate_variance_inflation_factor(df):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    vif["Feature"] = df.columns
    print(vif)
    return vif


# Split data into test and train.
def split_data(df, feature, test_size=0.2, random_state=None):
    y = df[feature]
    x = df.drop([feature], axis=1)
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return x_train_data, x_test_data, y_train_data, y_test_data


# Scale data using a scaler.
def scale_data(x_train_data, x_test_data):
    scaler = StandardScaler()
    x_train_data_scaled = scaler.fit_transform(x_train_data)
    x_test_data_scaled = scaler.transform(x_test_data)
    return x_train_data_scaled, x_test_data_scaled


def print_linear_regression_model_stats(x_test, y_test, yh):
    # print(calculate_variance_inflation_factor(preprocessed_data))
    print(f"Mean Absolute Error: {np.mean(np.abs(y_test - yh))}")

    mse = np.mean((y_test - yh) ** 2)
    print(f"Mean Squared Error: {mse}")

    r_squared = 1 - (
        np.sum((y_test - yh) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    )
    print(f"R-squared: {r_squared}")

    n = len(y_test)  # Number of observations
    p = x_test.shape[1]  # Number of predictors
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    print(f"Adjusted R-squared: {adjusted_r_squared}")


def print_logistic_regression_model_stats(x_test, y_test, yh):
    # Accuracy
    accuracy = np.mean(y_test == yh)

    # Precision
    true_positives = np.sum((yh == 1) & (y_test == 1))
    predicted_positives = np.sum(yh == 1)
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0

    # Recall
    actual_positives = np.sum(y_test == 1)
    recall = true_positives / actual_positives if actual_positives > 0 else 0

    # F1 Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    print(f"Model Statistics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def undersampling_dataset(df, target, test_size=0.2, random_state=None):
    positive = df[df[target] == 1]  # Extract all the true features in the dataset
    negative = df[df[target] == 0]  # Extract all the false features in the dataset

    # Select an equal number of negative examples as positive_train for training
    negative_sample = negative.sample(n=len(positive), random_state=random_state)
    df = pd.concat([positive, negative_sample])

    x_train, x_test, y_train, y_test = split_data(
        df, target, test_size=test_size, random_state=None
    )

    return x_train, x_test, y_train, y_test


def oversampling_dataset(df, target, test_size=0.2, random_state=None):
    positive = df[df[target] == 1]  # Extract all the true features in the dataset
    negative = df[df[target] == 0]  # Extract all the false features in the dataset

    # Sample with replacement from the positive examples to match the number of negatives
    positive_sample = positive.sample(
        n=len(negative), replace=True, random_state=random_state
    )
    df = pd.concat([negative, positive_sample])

    x_train, x_test, y_train, y_test = split_data(
        df, target, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def sigmoid(z):
    """
    Compute the sigmoid function for the input z.
    """
    return 1.0 / (1 + np.exp(-z))


def adaptive_moment_estimation(
    x_batch,
    y_batch,
    weights,
    m,
    v,
    t,
    learning_rate,
    beta1,
    beta2,
    adam_epsilon,
    gradient_fn,
):
    n_samples = x_batch.shape[0]

    for i in range(n_samples):  # Iterate over mini-batch
        rand_index = np.random.randint(n_samples)
        x_rand, y_rand = x_batch[rand_index], y_batch[rand_index]

        # Calculate gradient using the provided gradient function
        grad = gradient_fn(x_rand, y_rand)
        t += 1  # Increment time step

        # Update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)

        # Bias-corrected moment estimates
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update weights using Adam's update rule
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + adam_epsilon)

    return weights, m, v, t


def create_array_minibatch(x, y, batch_size):
    """
    Create mini-batches for mini-batch gradient descent.
    """
    # matrix = np.c_[x, y]
    # np.random.shuffle(matrix)
    # mini_batches = np.array_split(matrix, batch_size)
    # return [(batch[:, :-1], batch[:, -1]) for batch in mini_batches]

    matrix = np.c_[x, y]
    np.random.shuffle(matrix)
    mini_batches = []
    for i in range(0, len(matrix), batch_size):
        batch = matrix[i : i + batch_size]
        mini_batches.append((batch[:, :-1], batch[:, -1]))

    # mini_batches = np.array_split(matrix, batch_size)
    # return [(batch[:, :-1], batch[:, -1]) for batch in mini_batches]
    return mini_batches


def stochastic_gradient_descent(self, x, y):
    """
    Implement stochastic gradient descent.
    """
    grad = self.gradient(x, y)
    self.weights = self.weights - self.learning_rate * grad

