import os
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns


def plot_histogram(df, output_folder=None, filename="histogram.png"):
    df.hist(bins=10, figsize=(14, 14))
    plt.tight_layout()
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()
    else:
        plt.show()


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


def plot_logistic_learning_rates(results, learning_rates, output_folder="Results", title1="Task3.5_logistic_acc",
                                 title2="Task3.5_logistic_f1"
                                 , title3="Task3.5_logistic_pre", title4="Task3.5_logistic_rec"):
    # Plot MSE for both models
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["log_acc_train"], label="Accuracy Train Set", marker='o')
    plt.plot(learning_rates, results["log_acc_test"], label="Accuracy Test Set", marker='o')
    plt.plot(learning_rates, results["mini_acc_train"], label="MBSGD Accuracy Train Set", marker='o')
    plt.plot(learning_rates, results["mini_acc_test"], label="MBSGD Accuracy Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title1}.png")
    plt.savefig(plot_path)
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["log_f1_train"], label="f1 Train Set", marker='o')
    plt.plot(learning_rates, results["log_f1_test"], label="f1 Test Set", marker='o')
    plt.plot(learning_rates, results["mini_f1_train"], label="MBSGD f1 Train Set", marker='o')
    plt.plot(learning_rates, results["mini_f1_test"], label="MBSGD f1 Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("F-1 Score")
    plt.title("F-1 Score vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title2}.png")
    plt.savefig(plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["log_pre_train"], label="Precision Train Set", marker='o')
    plt.plot(learning_rates, results["log_pre_test"], label="Precision Test Set", marker='o')
    plt.plot(learning_rates, results["mini_pre_train"], label="MBSGD Precision Train Set", marker='o')
    plt.plot(learning_rates, results["mini_pre_test"], label="MBSGD Precision Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("Precision")
    plt.title("Precision vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title3}.png")
    plt.savefig(plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["log_rec_train"], label="Recall Train Set", marker='o')
    plt.plot(learning_rates, results["log_rec_test"], label="Recall Test Set", marker='o')
    plt.plot(learning_rates, results["mini_rec_train"], label="MBSGD Recall Train Set", marker='o')
    plt.plot(learning_rates, results["mini_rec_test"], label="MBSGD Recall Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("Recall")
    plt.title("Recall vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title4}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_mse_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_linear_mse"):
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
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_r2_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_linear_r2"):
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
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_evs_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_linear_evs"):
    # Plot evs for both models
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
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_acc_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_logistic_acc"):
    # Plot evs for both models
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["log_acc_train"], label="logistic acc Train Set", marker='o')
    plt.plot(train_sizes, results["log_acc_test"], label="logistic acc Test Set", marker='o')
    plt.plot(train_sizes, results["mini_acc_train"], label="MBSGD acc Train Set", marker='o')
    plt.plot(train_sizes, results["mini_acc_test"], label="MBSGD acc Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Training Set Size")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_f1_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_logistic_f1"):
    # Plot evs for both models
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["log_f1_train"], label="logistic f1 Train Set", marker='o')
    plt.plot(train_sizes, results["log_f1_test"], label="logistic f1 Test Set", marker='o')
    plt.plot(train_sizes, results["mini_f1_train"], label="MBSGD f1 Train Set", marker='o')
    plt.plot(train_sizes, results["mini_f1_test"], label="MBSGD f1 Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("F1-score")
    plt.title("F1-score vs Training Set Size")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_pre_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_logistic_pre"):
    # Plot evs for both models
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["log_pre_train"], label="logistic pre Train Set", marker='o')
    plt.plot(train_sizes, results["log_pre_test"], label="logistic pre Test Set", marker='o')
    plt.plot(train_sizes, results["mini_pre_train"], label="MBSGD pre Train Set", marker='o')
    plt.plot(train_sizes, results["mini_pre_test"], label="MBSGD pre Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Precision")
    plt.title("Precision vs Training Set Size")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_rec_growing_set(results, train_sizes, output_folder="Results", title="Task3.3_logistic_rec"):
    # Plot evs for both models
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, results["log_rec_train"], label="logistic rec Train Set", marker='o')
    plt.plot(train_sizes, results["log_rec_test"], label="logistic rec Test Set", marker='o')
    plt.plot(train_sizes, results["mini_rec_train"], label="MBSGD rec Train Set", marker='o')
    plt.plot(train_sizes, results["mini_rec_test"], label="MBSGD rec Test Set", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Recall")
    plt.title("Recall vs Training Set Size")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_batch_sizes_result(losses_by_batch_size, r2s_by_batch_size, log=0,
                            output_folder="Results", title1="Task3.4_linear_mse",
                            title2="Task3.4_linear_r2"):
    # Plotting the loss vs iterations for different batch sizes
    plt.figure(figsize=(10, 6))
    if log == 0:
        for batch_size, losses in losses_by_batch_size.items():
            plt.plot(range(len(losses)), losses, label=f'Batch Size {batch_size}')

        plt.xlabel('Iterations')
        plt.ylabel('Training Loss')
        plt.title('Linear Training Loss vs Number of Iterations for Different Batch Sizes')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plot_path = os.path.join(output_folder, f"{title1}.png")
        plt.savefig(plot_path)
        plt.close()
    else:
        for batch_size, losses in losses_by_batch_size.items():
            plt.plot(range(len(losses)), losses, label=f'Batch Size {batch_size}')

        plt.xlabel('Iterations')
        plt.ylabel('Training Loss')
        plt.title('Logistic Training Loss vs Number of Iterations for Different Batch Sizes')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plot_path = os.path.join(output_folder, f"{title1}.png")
        plt.savefig(plot_path)
        plt.close()

    plt.figure(figsize=(10, 6))
    if log == 0:
        for batch_size, r2s in r2s_by_batch_size.items():
            plt.plot(range(len(r2s)), r2s, label=f'Batch Size {batch_size}')

        plt.xlabel('Iterations')
        plt.ylabel('R² Score')
        plt.title('R² Score vs Number of Iterations for Different Batch Sizes')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_folder, f"{title2}.png")
        plt.savefig(plot_path)
        plt.close()


def plot_linear_learning_rates(results, learning_rates, output_folder="Results", title1="Task3.5_linear_mse",
                               title2="Task3.5_linear_R2"):
    # Plot MSE for both models
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["mini_mse_train"], label="MSE Train Set", marker='o')
    plt.plot(learning_rates, results["mini_mse_test"], label="MSE Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title1}.png")
    plt.savefig(plot_path)
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, results["mini_r2_train"], label="R Square Train Set", marker='o')
    plt.plot(learning_rates, results["mini_r2_test"], label="R square Test Set", marker='o')
    plt.xlabel("Learning Rates")
    plt.ylabel("R Square")
    plt.title("R2 vs Learning Rates")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_folder, f"{title2}.png")
    plt.savefig(plot_path)
    plt.close()
