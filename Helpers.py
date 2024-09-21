import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
    return vif


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
def split_data(df_thermography, feature, test_size=0.2, random_state=None):
    y = df_thermography[feature]
    x = df_thermography.drop([feature], axis=1)
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


def plot_actual_vs_predicted(y_test, y_hat, target_name, output_folder):
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

    plot_path = os.path.join(
        output_folder, f"Graph_Actual_vs_Predicted Value_Linear_Regression.png"
    )
    plt.savefig(plot_path)
    plt.close()
