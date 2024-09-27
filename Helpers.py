import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,mean_squared_error, r2_score, explained_variance_score



def plot_histogram(df, output_folder=None, filename="histogram.png"):
    df.hist(bins=10, figsize=(14, 14))
    plt.tight_layout()
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()
    else:
        plt.show()


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


def plot_variance_inflation_factor(
        df, target, output_folder=None, filename="variance_inflation_factor.png"
):
    # Calculate the VIF for each feature
    vif = pd.DataFrame()
    vif["VIF Factor"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    vif["Feature"] = df.columns

    # Sort the VIF values
    vif_sorted = vif.sort_values(by="VIF Factor", ascending=False)

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))
    ax = vif_sorted.plot(
        kind="bar", x="Feature", y="VIF Factor", legend=False, color="skyblue"
    )
    plt.grid(True, which="both", axis="y", linestyle="--")

    # Annotate bars with VIF values
    for idx, value in enumerate(vif_sorted["VIF Factor"]):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom")

    # Set titles and labels
    plt.title(
        f"Variance Inflation Factor (VIF) for Features\n(Target: {target})", fontsize=14
    )
    plt.ylabel("VIF Factor", fontsize=12)
    plt.xlabel("Features", fontsize=12)

    plt.xticks(rotation=45)

    # Save the plot if output_folder is specified
    if output_folder:
        plt.tight_layout()
        plot_path = os.path.join(output_folder, filename)
        plt.savefig(plot_path)
    plt.close()


# Plot all correlations.
def compute_and_plot_correlation(df, target, output_folder, filename="correlation.png"):
    correlations = df.corr()[target].drop(target)

    # Create a bar plot for the correlations
    plt.figure(figsize=(12, 8))
    ax = correlations.sort_values(ascending=False).plot(kind="bar")

    plt.grid(True, which="both", axis="y", linestyle="--")

    for idx, value in enumerate(correlations.sort_values(ascending=False)):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom")

    plt.title(f"Correlation of Features with {target}", fontsize=14)
    plt.ylabel("Correlation Coefficient", fontsize=12)
    plt.xlabel("Features", fontsize=12)

    plt.xticks(rotation=90)

    plt.tight_layout()
    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()


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


def plot_actual_vs_predicted(
        y_test,
        y_hat,
        target_name,
        output_folder,
        filename="Graph_Actual_vs_Predicted Value_Linear_Regression.png",
):
    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_test,
        y_hat,
        alpha=0.5,
        color="blue",
        label=f"Predicted vs. Actual of {target_name}",
    )
    plt.xlabel(f"Actual Values of {target_name}")
    plt.ylabel(f"Predicted Values of {target_name}")
    plt.title(f"Actual vs. Predicted Values of {target_name}")

    # Prediction line
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="red",
        linestyle="--",
        label=f"Perfect Prediction Line of {target_name}",
    )

    plt.legend()

    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()


def plot_residual(y_test, yh, output_folder="Results", filename="Residual.png"):
    residuals = y_test - yh
    plt.scatter(yh, residuals)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")

    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()


def plot_residual_distribution(
        y_test, yh, output_folder="Results", filename="Residual_Distribution.png"
):
    residuals = y_test - yh
    plt.hist(residuals, bins=30)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()


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


def plot_correlation_matrix(X, output_folder="Results", title="Correlation Matrix"):
    plt.figure(figsize=(20, 20))
    heatmap = sns.heatmap(X.corr(), vmin=-1, vmax=1, annot=True, cmap="Blues")
    heatmap.set_title("Correlation heatmap", fontdict={"fontsize": 10}, pad=12)
    plot_path = os.path.join(output_folder, title)
    plt.savefig(plot_path)
    plt.close()


def plot_confusion_matrix(
        y_true, y_hat, output_folder="Results", title="Confusion Matrix"
):
    cm = confusion_matrix(y_true, y_hat)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["False", "True"])
    plt.yticks(tick_marks, ["False", "True"])

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_roc_curve(y_true, y_hat, output_folder="Results", title="ROC_Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_hat)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


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
        x_batch, y_batch, weights, m, v, t, learning_rate, beta1, beta2, adam_epsilon, gradient_fn
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
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # Bias-corrected moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update weights using Adam's update rule
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + adam_epsilon)

    return weights, m, v, t


def create_array_minibatch(x, y, batch_size):
    """
    Create mini-batches for mini-batch gradient descent.
    """
    matrix = np.c_[x, y]
    np.random.shuffle(matrix)
    mini_batches = np.array_split(matrix, batch_size)
    return [(batch[:, :-1], batch[:, -1]) for batch in mini_batches]


def stochastic_gradient_descent(self, x, y):
    """
      Implement stochastic gradient descent.
    """
    n_samples, n_features = x.shape

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
def plot_growing_subset_linear(X_train,y_train,X_test,y_test):
    train_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        """Helper function to fit the model and calculate metrics."""
        # Fit the model
        model.fit(X_train, y_train)
    
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
        # Compute MSE and R² for both train and test
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_evs = explained_variance_score(y_train, y_pred_train)
        test_evs = explained_variance_score(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
    
        return train_mse, test_mse, train_r2, test_r2 , train_evs, test_evs
    results = {
    "lin_mse_train": [],
    "lin_mse_test": [],
    "lin_r2_train": [],
    "lin_r2_test": [],
    "lin_evs_train": [],
    "lin_evs_test": [],
    "mini_mse_train": [],
    "mini_mse_test": [],
    "mini_r2_train": [],
    "mini_r2_test": [],
    "mini_evs_train": [],
    "mini_evs_test": []
    }

    for size in train_sizes:
        # Sample a subset of the training data
        X_train_subset = X_train.sample(frac=size, random_state=3)
        y_train_subset = y_train.loc[X_train_subset.index]

    # Analytical Linear Regression
        analytical = LinearRegression()
        lin_mse_train, lin_mse_test, lin_r2_train, lin_r2_test, lin_evs_train,lin_evs_test= evaluate_model(
            analytical, X_train_subset, y_train_subset, X_test, y_test
        )
    
    # Store results for analytical model
        results["lin_mse_train"].append(lin_mse_train)
        results["lin_mse_test"].append(lin_mse_test)
        results["lin_r2_train"].append(lin_r2_train)
        results["lin_r2_test"].append(lin_r2_test)
        results["lin_evs_train"].append(lin_evs_train)
        results["lin_evs_test"].append(lin_evs_test)

    # Mini-Batch SGD Linear Regression
        minibatch = MiniBatchStochasticLinearRegression(learning_rate=0.05, batch_size=32, epochs=1000)
        mini_mse_train, mini_mse_test, mini_r2_train, mini_r2_test, mini_evs_train, mini_evs_test = evaluate_model(
            minibatch, X_train_subset, y_train_subset, X_test, y_test
        )
    
    # Store results for MBSGD model
    results["mini_mse_train"].append(mini_mse_train)
    results["mini_mse_test"].append(mini_mse_test)
    results["mini_r2_train"].append(mini_r2_train)
    results["mini_r2_test"].append(mini_r2_test)
    results["mini_evs_train"].append(mini_evs_train)
    results["mini_evs_test"].append(mini_evs_test)

    # Plot MSE for both models
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["lin_mse_train"], label="Analytical MSE Train Set", marker='o')
    plt.plot(train_sizes, results["lin_mse_test"], label="Analytical MSE Test Set", marker='o')
    plt.plot(train_sizes, results["mini_mse_train"], label="MBSGD MSE Train Set", marker='o')
    plt.plot(train_sizes, results["mini_mse_test"], label="MBSGD MSE Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Squared Error")
    plt.title("Mean Squared Error vs Training Set Size")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot R² for both models
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["lin_r2_train"], label="Analytical R² Train Set", marker='o')
    plt.plot(train_sizes, results["lin_r2_test"], label="Analytical R² Test Set", marker='o')
    plt.plot(train_sizes, results["mini_r2_train"], label="MBSGD R² Train Set", marker='o')
    plt.plot(train_sizes, results["mini_r2_test"], label="MBSGD R² Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("R² Score")
    plt.title("R² Score vs Training Set Size")
    plt.legend()
    plt.grid(True)
    plt.show()

    #Plot evs for both models
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["lin_evs_train"], label="Analytical evs Train Set", marker='o')
    plt.plot(train_sizes, results["lin_evs_test"], label="Analytical evs Test Set", marker='o')
    plt.plot(train_sizes, results["mini_evs_train"], label="MBSGD evs Train Set", marker='o')
    plt.plot(train_sizes, results["mini_evs_test"], label="MBSGD evs Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Explained Variance Score")
    plt.title("Explained Variance Score vs Training Set Size")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def linear_batch_size_test():
    

    