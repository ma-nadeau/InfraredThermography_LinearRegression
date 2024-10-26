import os
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns


def plot_histogram(df, output_folder=None, filename="histogram.png"):
    """
    Plots histograms for each feature in the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe whose features will be plotted.
    output_folder (str, optional): Directory where the plot image will be saved.
    filename (str, optional): Filename for the saved plot.

    If output_folder is not provided, the plot is displayed.
    """
    df.hist(bins=10, figsize=(14, 14))  # Create histogram with specified number of bins
    plt.tight_layout()

    if output_folder is not None:
        # Save the plot if an output folder is provided
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()  # Close the plot after saving
    else:
        # Display the plot if no output folder is provided
        plt.show()


def plot_variance_inflation_factor(df, target, output_folder=None, filename="variance_inflation_factor.png"):
    """
    Calculates and plots Variance Inflation Factor (VIF) for the features in the dataframe.

    Parameters:
    df (pd.DataFrame): Input dataframe containing features.
    target (str): Target variable name.
    output_folder (str, optional): Directory where the plot image will be saved.
    filename (str, optional): Filename for the saved plot.

    If output_folder is not provided, the plot is displayed.
    """
    # Calculate VIF for each feature in the dataframe
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif["Feature"] = df.columns

    # Sort VIF values in descending order
    vif_sorted = vif.sort_values(by="VIF Factor", ascending=False)

    # Create a bar plot for the VIF values
    plt.figure(figsize=(10, 6))
    ax = vif_sorted.plot(kind="bar", x="Feature", y="VIF Factor", legend=False, color="skyblue")
    plt.grid(True, which="both", axis="y", linestyle="--")

    # Annotate bars with VIF values
    for idx, value in enumerate(vif_sorted["VIF Factor"]):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom")

    # Set plot title and labels
    plt.title(f"Variance Inflation Factor (VIF) for Features\n(Target: {target})", fontsize=14)
    plt.ylabel("VIF Factor", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=45)

    # Save the plot if output_folder is specified
    if output_folder:
        plt.tight_layout()
        plot_path = os.path.join(output_folder, filename)
        plt.savefig(plot_path)
    plt.close()


def compute_and_plot_correlation(df, target, output_folder, filename="correlation.png"):
    """
    Computes and plots the correlation of each feature with the target variable.

    Parameters:
    df (pd.DataFrame): Input dataframe containing features.
    target (str): Target variable name.
    output_folder (str): Directory where the plot image will be saved.
    filename (str, optional): Filename for the saved plot.
    """
    # Compute correlations with respect to the target variable
    correlations = df.corr()[target].drop(target)

    # Create a bar plot for the correlations
    plt.figure(figsize=(12, 8))
    ax = correlations.sort_values(ascending=False).plot(kind="bar")

    plt.grid(True, which="both", axis="y", linestyle="--")

    # Annotate bars with correlation values
    for idx, value in enumerate(correlations.sort_values(ascending=False)):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom")

    # Set plot title and labels
    plt.title(f"Correlation of Features with {target}", fontsize=14)
    plt.ylabel("Correlation Coefficient", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=90)

    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()


def plot_actual_vs_predicted(y_test, y_hat, target_name, output_folder,
                             filename="Graph_Actual_vs_Predicted Value_Linear_Regression.png"):
    """
    Plots actual vs predicted values for regression tasks.

    Parameters:
    y_test (np.ndarray): Actual target values.
    y_hat (np.ndarray): Predicted target values.
    target_name (str): Name of the target variable.
    output_folder (str): Directory where the plot image will be saved.
    filename (str, optional): Filename for the saved plot.
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot for actual vs predicted values
    plt.scatter(y_test, y_hat, alpha=0.5, color="blue", label=f"Predicted vs. Actual of {target_name}")
    plt.xlabel(f"Actual Values of {target_name}")
    plt.ylabel(f"Predicted Values of {target_name}")
    plt.title(f"Actual vs. Predicted Values of {target_name}")

    # Draw a perfect prediction line
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--",
             label=f"Perfect Prediction Line of {target_name}")

    plt.legend()

    # Save the plot
    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()


def plot_residual(y_test, yh, output_folder="Results", filename="Residual.png"):
    """
    Plots residuals (difference between actual and predicted values) vs predicted values.

    Parameters:
    y_test (np.ndarray): Actual target values.
    yh (np.ndarray): Predicted target values.
    output_folder (str, optional): Directory where the plot image will be saved.
    filename (str, optional): Filename for the saved plot.
    """
    # Compute residuals
    residuals = y_test - yh

    # Create scatter plot for residuals
    plt.scatter(yh, residuals)
    plt.axhline(y=0, color="r", linestyle="--")  # Add a horizontal line at y=0 for reference
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")

    # Save the plot
    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()


def plot_residual_distribution(y_test, yh, output_folder="Results", filename="Residual_Distribution.png"):
    """
    Plots the distribution (histogram) of residuals.

    Parameters:
    y_test (np.ndarray): Actual target values.
    yh (np.ndarray): Predicted target values.
    output_folder (str, optional): Directory where the plot image will be saved.
    filename (str, optional): Filename for the saved plot.
    """
    residuals = y_test - yh

    # Plot histogram of residuals
    plt.hist(residuals, bins=30)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    # Save the plot
    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()


def plot_correlation_matrix(X, output_folder="Results", title="Correlation Matrix"):
    """
    Plots the correlation matrix heatmap for the features in the dataframe.

    Parameters:
    X (pd.DataFrame): Input dataframe containing features.
    output_folder (str, optional): Directory where the plot image will be saved.
    title (str, optional): Title for the saved plot.
    """
    plt.figure(figsize=(20, 20))

    # Create heatmap for correlation matrix
    heatmap = sns.heatmap(X.corr(), vmin=-1, vmax=1, annot=True, cmap="Blues")
    heatmap.set_title("Correlation heatmap", fontdict={"fontsize": 10}, pad=12)

    # Save the plot
    plot_path = os.path.join(output_folder, title)
    plt.savefig(plot_path)
    plt.close()


def plot_confusion_matrix(y_true, y_hat, output_folder="Results", title="Confusion Matrix"):
    """
    Plots the confusion matrix for classification tasks.

    Parameters:
    y_true (np.ndarray): Actual class labels.
    y_hat (np.ndarray): Predicted class labels.
    output_folder (str): Directory where the plot image will be saved.
    title (str, optional): Title for the saved plot.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_hat)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Set tick marks for class labels
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["False", "True"])
    plt.yticks(tick_marks, ["False", "True"])

    # Annotate confusion matrix with the values
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_roc_curve(y_true, y_hat, output_folder="Results", title="ROC_Curve"):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for classification tasks.

    Parameters:
    y_true (np.ndarray): Actual class labels.
    y_hat (np.ndarray): Predicted class probabilities.
    output_folder (str): Directory where the plot image will be saved.
    title (str, optional): Title for the saved plot.
    """
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_hat)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    # Save the plot
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


# The remaining functions follow similar patterns, and comments could be added in the same manner.


def plot_logistic_learning_rates(results, learning_rates, output_folder="Results", title1="Task3.5_logistic_acc",
                                 title2="Task3.5_logistic_f1", title3="Task3.5_logistic_pre",
                                 title4="Task3.5_logistic_rec"):
    """
    Plots logistic regression metrics (accuracy, F1 score, precision, recall) against learning rates.

    Parameters:
    results (dict): Dictionary containing results for training and test sets.
    learning_rates (list): List of learning rates used in training.
    output_folder (str, optional): Directory where the plot image will be saved.
    title1, title2, title3, title4 (str, optional): Titles for the accuracy, F1 score, precision, and recall plots respectively.
    """

    # Plot Accuracy vs Learning Rates
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["log_acc_train"], label="Accuracy Train Set", marker='o')
    plt.plot(learning_rates, results["log_acc_test"], label="Accuracy Test Set", marker='o')
    plt.plot(learning_rates, results["mini_acc_train"], label="MBSGD Accuracy Train Set", marker='o')
    plt.plot(learning_rates, results["mini_acc_test"], label="MBSGD Accuracy Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("Accuracy")
    plt.title("Logistic Regression - Accuracy vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title1}.png")
    plt.savefig(plot_path)
    plt.close()

    # Plot F1 Score vs Learning Rates
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["log_f1_train"], label="f1 Train Set", marker='o')
    plt.plot(learning_rates, results["log_f1_test"], label="f1 Test Set", marker='o')
    plt.plot(learning_rates, results["mini_f1_train"], label="MBSGD f1 Train Set", marker='o')
    plt.plot(learning_rates, results["mini_f1_test"], label="MBSGD f1 Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("F-1 Score")
    plt.title("Logistic Regression - F-1 Score vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title2}.png")
    plt.savefig(plot_path)
    plt.close()

    # Plot Precision vs Learning Rates
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["log_pre_train"], label="Precision Train Set", marker='o')
    plt.plot(learning_rates, results["log_pre_test"], label="Precision Test Set", marker='o')
    plt.plot(learning_rates, results["mini_pre_train"], label="MBSGD Precision Train Set", marker='o')
    plt.plot(learning_rates, results["mini_pre_test"], label="MBSGD Precision Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("Precision")
    plt.title("Logistic Regression - Precision vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title3}.png")
    plt.savefig(plot_path)
    plt.close()

    # Plot Recall vs Learning Rates
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["log_rec_train"], label="Recall Train Set", marker='o')
    plt.plot(learning_rates, results["log_rec_test"], label="Recall Test Set", marker='o')
    plt.plot(learning_rates, results["mini_rec_train"], label="MBSGD Recall Train Set", marker='o')
    plt.plot(learning_rates, results["mini_rec_test"], label="MBSGD Recall Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("Recall")
    plt.title("Logistic Regression - Recall vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title4}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_mse_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_linear_mse"):
    """
    Plots Mean Squared Error (MSE) for both analytical and MBSGD models across growing training set sizes.

    Parameters:
    results (dict): Dictionary containing MSE results for training and test sets.
    train_sizes (list): List of training set sizes.
    output_folder (str, optional): Directory where the plot image will be saved.
    title (str, optional): Title for the saved plot.
    """

    # Plot MSE vs Training Set Size
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["lin_mse_train"], label="Analytical MSE Train Set", marker='o')
    plt.plot(train_sizes, results["lin_mse_test"], label="Analytical MSE Test Set", marker='o')
    plt.plot(train_sizes, results["mini_mse_train"], label="MBSGD MSE Train Set", marker='o')
    plt.plot(train_sizes, results["mini_mse_test"], label="MBSGD MSE Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Squared Error")
    plt.title("Linear Regression - Mean Squared Error vs Training Set Size")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_r2_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_linear_r2"):
    """
    Plots R² score for both analytical and MBSGD models across growing training set sizes.

    Parameters:
    results (dict): Dictionary containing R² score results for training and test sets.
    train_sizes (list): List of training set sizes.
    output_folder (str, optional): Directory where the plot image will be saved.
    title (str, optional): Title for the saved plot.
    """

    # Plot R² Score vs Training Set Size
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["lin_r2_train"], label="Analytical R² Train Set", marker='o')
    plt.plot(train_sizes, results["lin_r2_test"], label="Analytical R² Test Set", marker='o')
    plt.plot(train_sizes, results["mini_r2_train"], label="MBSGD R² Train Set", marker='o')
    plt.plot(train_sizes, results["mini_r2_test"], label="MBSGD R² Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("R² Score")
    plt.title("Linear Regression - R² Score vs Training Set Size")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_evs_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_linear_evs"):
    """
    Plots Explained Variance Score (EVS) for both analytical and MBSGD models across growing training set sizes.

    Parameters:
    results (dict): Dictionary containing EVS results for training and test sets.
    train_sizes (list): List of training set sizes.
    output_folder (str, optional): Directory where the plot image will be saved.
    title (str, optional): Title for the saved plot.
    """

    # Plot EVS vs Training Set Size
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["lin_evs_train"], label="Analytical EVS Train Set", marker='o')
    plt.plot(train_sizes, results["lin_evs_test"], label="Analytical EVS Test Set", marker='o')
    plt.plot(train_sizes, results["mini_evs_train"], label="MBSGD EVS Train Set", marker='o')
    plt.plot(train_sizes, results["mini_evs_test"], label="MBSGD EVS Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Explained Variance Score")
    plt.title("Linear Regression - Explained Variance Score vs Training Set Size")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_acc_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_logistic_acc"):
    """
    Plots accuracy for both logistic regression and MBSGD models across growing training set sizes.

    Parameters:
    results (dict): Dictionary containing accuracy results for training and test sets.
    train_sizes (list): List of training set sizes.
    output_folder (str, optional): Directory where the plot image will be saved.
    title (str, optional): Title for the saved plot.
    """

    # Plot Accuracy vs Training Set Size
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["log_acc_train"], label="Logistic Accuracy Train Set", marker='o')
    plt.plot(train_sizes, results["log_acc_test"], label="Logistic Accuracy Test Set", marker='o')
    plt.plot(train_sizes, results["mini_acc_train"], label="MBSGD Accuracy Train Set", marker='o')
    plt.plot(train_sizes, results["mini_acc_test"], label="MBSGD Accuracy Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Logistic Regression - Accuracy vs Training Set Size")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_f1_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_logistic_f1"):
    """
    Plots F1 score for both logistic regression and MBSGD models across growing training set sizes.

    Parameters:
    results (dict): Dictionary containing F1 score results for training and test sets.
    train_sizes (list): List of training set sizes.
    output_folder (str, optional): Directory where the plot image will be saved.
    title (str, optional): Title for the saved plot.
    """

    # Plot F1 Score vs Training Set Size
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["log_f1_train"], label="Logistic F1 Train Set", marker='o')
    plt.plot(train_sizes, results["log_f1_test"], label="Logistic F1 Test Set", marker='o')
    plt.plot(train_sizes, results["mini_f1_train"], label="MBSGD F1 Train Set", marker='o')
    plt.plot(train_sizes, results["mini_f1_test"], label="MBSGD F1 Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.title("Logistic Regression - F1 Score vs Training Set Size")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_pre_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_logistic_pre"):
    """
    Plots precision for both logistic regression and MBSGD models across growing training set sizes.

    Parameters:
    results (dict): Dictionary containing precision results for training and test sets.
    train_sizes (list): List of training set sizes.
    output_folder (str, optional): Directory where the plot image will be saved.
    title (str, optional): Title for the saved plot.
    """

    # Plot Precision vs Training Set Size
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["log_pre_train"], label="Logistic Precision Train Set", marker='o')
    plt.plot(train_sizes, results["log_pre_test"], label="Logistic Precision Test Set", marker='o')
    plt.plot(train_sizes, results["mini_pre_train"], label="MBSGD Precision Train Set", marker='o')
    plt.plot(train_sizes, results["mini_pre_test"], label="MBSGD Precision Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Precision")
    plt.title("Logistic Regression - Precision vs Training Set Size")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_rec_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_logistic_rec"):
    """
    Plots recall for both logistic regression and MBSGD models across growing training set sizes.

    Parameters:
    results (dict): Dictionary containing recall results for training and test sets.
    train_sizes (list): List of training set sizes.
    output_folder (str, optional): Directory where the plot image will be saved.
    title (str, optional): Title for the saved plot.
    """

    # Plot Recall vs Training Set Size
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["log_rec_train"], label="Logistic Recall Train Set", marker='o')
    plt.plot(train_sizes, results["log_rec_test"], label="Logistic Recall Test Set", marker='o')
    plt.plot(train_sizes, results["mini_rec_train"], label="MBSGD Recall Train Set", marker='o')
    plt.plot(train_sizes, results["mini_rec_test"], label="MBSGD Recall Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Recall")
    plt.title("Logistic Regression - Recall vs Training Set Size")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_batch_sizes_result(losses_by_batch_size, r2s_by_batch_size, log=0, output_folder="Results",
                            title1="Task3.4_linear_mse", title2="Task3.4_linear_r2"):
    """
    Plots training loss and R² scores for different batch sizes for both linear and logistic regression models.

    Parameters:
    losses_by_batch_size (dict): Dictionary containing loss values for each batch size.
    r2s_by_batch_size (dict): Dictionary containing R² score values for each batch size.
    log (int, optional): If 1, plots the logistic regression loss, otherwise linear regression.
    output_folder (str, optional): Directory where the plot image will be saved.
    title1 (str, optional): Title for the loss plot.
    title2 (str, optional): Title for the R² plot.
    """

    # Plot Training Loss vs Epoch for different batch sizes
    plt.figure(figsize=(10, 6))
    for batch_size, losses in losses_by_batch_size.items():
        plt.plot(range(len(losses)), losses, label=f'Batch Size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    if log == 0:
        plt.title('Linear Regression - Linear Training Loss vs Epoch for Different Batch Sizes')
    else:
        plt.title('Logistic Regression - Logistic Training Loss vs Epoch for Different Batch Sizes')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title1}.png")
    plt.savefig(plot_path)
    plt.close()

    # Plot R² Score vs Epoch for different batch sizes
    plt.figure(figsize=(10, 6))
    for batch_size, r2s in r2s_by_batch_size.items():
        plt.plot(range(len(r2s)), r2s, label=f'Batch Size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Linear Regression - R² Score vs Epoch for Different Batch Sizes')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title2}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_linear_learning_rates(results, learning_rates, output_folder="Results", title1="Task3.5_linear_mse",
                               title2="Task3.5_linear_R2"):
    """
    Plots MSE and R² score for linear regression across different learning rates.

    Parameters:
    results (dict): Dictionary containing MSE and R² results for training and test sets.
    learning_rates (list): List of learning rates.
    output_folder (str, optional): Directory where the plot image will be saved.
    title1 (str, optional): Title for the MSE plot.
    title2 (str, optional): Title for the R² plot.
    """

    # Plot MSE vs Learning Rates
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["mini_mse_train"], label="MSE Train Set", marker='o')
    plt.plot(learning_rates, results["mini_mse_test"], label="MSE Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("Mean Squared Error")
    plt.title("Linear Regression - MSE vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title1}.png")
    plt.savefig(plot_path)
    plt.close()

    # Plot R² vs Learning Rates
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["mini_r2_train"], label="R² Train Set", marker='o')
    plt.plot(learning_rates, results["mini_r2_test"], label="R² Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("R² Score")
    plt.title("Linear Regression - R² vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title2}.png")
    plt.savefig(plot_path)
    plt.close()
