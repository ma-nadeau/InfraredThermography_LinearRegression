# Plot histogram for each feature.
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
