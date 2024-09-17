import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


def preprocess_thermography_data(file_name):
    # Load the thermography dataset into a Pandas dataframe.
    df_thermography = pd.read_csv(file_name)

    # Initialize encoder and scaler.
    label_encoder = LabelEncoder()

    # Remove SubjectID column once validated for no duplicates.
    df_thermography = df_thermography.drop_duplicates(subset='SubjectID', keep='first')
    df_thermography = df_thermography.drop('SubjectID', axis=1)

    # Obtain midpoint value for integers in ranges.
    def get_midpoint(age_range):
        if '-' in age_range:
            lower, upper = age_range.split('-')
            return (int(lower) + int(upper)) / 2
        else:
            lower = int(age_range[1:])  # Extract the number after ">"
            return lower + 5  # Estimate of the population over 60 years.

    # Apply the midpoint function to the 'Age' column.
    df_thermography['Age'] = df_thermography['Age'].apply(get_midpoint)

    # Remove outliers in data using the Inter-quartile Range Method.
    def remove_outliers(df, column_name):
        # Compute Q1 (25th percentile) and Q3 (75th percentile)
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)

        # Compute the IQR
        iqr = q3 - q1

        # Define lower and upper bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Remove outliers
        df.drop(df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)].index, inplace=True)

    remove_outliers(df_thermography, 'Age')
    remove_outliers(df_thermography, 'Distance')
    remove_outliers(df_thermography, 'Humidity')

    # Apply LabelEncoder on the Gender and Ethnicity categorical columns.
    df_thermography['Gender'] = label_encoder.fit_transform(df_thermography['Gender'])
    df_thermography['Ethnicity'] = label_encoder.fit_transform(df_thermography['Ethnicity'])

    # Delete rows with missing values (entries in the 'Distance' feature column).
    df_thermography = df_thermography.fillna(df_thermography.mean())

    return df_thermography


def split_data(df_thermography):
    y = df_thermography['aveOralM']
    x = df_thermography.drop(['aveOralM'], axis=1)
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train_data, x_test_data, y_train_data, y_test_data


def scale_data(x_train_data, x_test_data):
    scaler = StandardScaler()
    x_train_data_scaled = scaler.fit_transform(x_train_data)
    x_test_data_scaled = scaler.transform(x_test_data)
    return x_train_data_scaled, x_test_data_scaled


# Plot histogram for each feature.
def plot_histogram(df, bins=10, figsize=(14,14)):
    df.hist(bins=10, figsize=(14, 14))
    plt.tight_layout()
    plt.show()


# Obtain correlation matrix.
def get_correlation(df, feature):
    correlation_matrix = df.corr()
    correlation = correlation_matrix[feature]
    return correlation


# Obtain Variance Inflation Factor (VIF)
def calculate_variance_inflation_factor(df):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif["Feature"] = df.columns
    return vif


preprocessed_data = preprocess_thermography_data('./InfraredThermographyTemperature.csv')
x_train, x_test, y_train, y_test = split_data(preprocessed_data)
x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

plot_histogram(preprocessed_data)
get_correlation(preprocessed_data, 'aveOralM')
calculate_variance_inflation_factor(preprocessed_data)