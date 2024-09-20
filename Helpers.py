import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_histogram(df):
    df.hist(bins=10, figsize=(14, 14))
    plt.tight_layout()
    plt.show()


# Obtain correlation matrix.
def get_correlation(df, target):
    correlation_matrix = df.corr()
    correlation = correlation_matrix[target]
    return correlation


# Obtain Variance Inflation Factor (VIF)
def calculate_variance_inflation_factor(df):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif["Feature"] = df.columns
    return vif


# Plot all correlations.
def compute_and_plot_correlation(df, output_folder):
    correlations = df.corr()["aveOralM"].drop("aveOralM")

    # Create a bar plot for the correlations
    plt.figure(figsize=(12, 8))
    ax = correlations.sort_values(ascending=False).plot(kind="bar")

    plt.grid(True, which="both", axis="y", linestyle="--")

    for idx, value in enumerate(correlations.sort_values(ascending=False)):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom")

    plt.title("Correlation of Features with aveOralM", fontsize=14)
    plt.ylabel("Correlation Coefficient", fontsize=12)
    plt.xlabel("Features", fontsize=12)

    plt.xticks(rotation=90)

    plt.tight_layout()
    plot_path = os.path.join(
        output_folder, f"Correlation_of_Features_with_aveOralM.png"
    )
    plt.savefig(plot_path)
    plt.close()


# Split data into test and train.
def split_data(df_thermography, feature):
    y = df_thermography[feature]
    x = df_thermography.drop([feature], axis=1)
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train_data, x_test_data, y_train_data, y_test_data


# Scale data using a scaler.
def scale_data(x_train_data, x_test_data):
    scaler = StandardScaler()
    x_train_data_scaled = scaler.fit_transform(x_train_data)
    x_test_data_scaled = scaler.transform(x_test_data)
    return x_train_data_scaled, x_test_data_scaled
