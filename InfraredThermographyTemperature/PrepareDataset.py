from sklearn.preprocessing import LabelEncoder

from Assignment1.Helpers import *
from MiniBatchStochasticLinearRegression import *
from LinearRegression import *


def preprocess_thermography_data(file_name):
    # Load the thermography dataset into a Pandas dataframe.
    df_thermography = pd.read_csv(file_name)

    # Initialize encoder and scaler.
    label_encoder = LabelEncoder()

    # Remove SubjectID column once validated for no duplicates.
    df_thermography = df_thermography.drop_duplicates(subset="SubjectID", keep="first")
    df_thermography = df_thermography.drop("SubjectID", axis=1)

    # Obtain midpoint value for integers in ranges.
    def get_midpoint(age_range):
        if "-" in age_range:
            lower, upper = age_range.split("-")
            return (int(lower) + int(upper)) / 2
        else:
            lower = int(age_range[1:])  # Extract the number after ">"
            return lower + 5  # Estimate of the population over 60 years.

    # Apply the midpoint function to the 'Age' column.
    df_thermography["Age"] = df_thermography["Age"].apply(get_midpoint)

    # Remove outliers in data using the Inter-quartile Range Method.
    def remove_outliers(df, column_name):
        # Compute Q1 (25th percentile) and Q3 (75th percentile)
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)

        # Compute the IQR
        iqr = q3 - q1

        # Define lower and upper bounds
        lower_bound = q1 - 2.5 * iqr
        upper_bound = q3 + 2.5 * iqr

        # Remove outliers
        df.drop(
            df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)].index,
            inplace=True,
        )

    remove_outliers(df_thermography, "Age")
    remove_outliers(df_thermography, "Distance")
    remove_outliers(df_thermography, "Humidity")

    # Apply LabelEncoder on the Gender and Ethnicity categorical columns.
    df_thermography["Gender"] = label_encoder.fit_transform(df_thermography["Gender"])
    df_thermography["Ethnicity"] = label_encoder.fit_transform(
        df_thermography["Ethnicity"]
    )

    # Replace empty values with mean of values in that column.
    df_thermography = df_thermography.fillna(df_thermography.mean())

    return df_thermography


def perform_plotting(df_thermography):

    plot_histogram(
        df_thermography,
        "Results",
        "Histogram_Infrared_Thermography_Temperature.png",
    )
    compute_and_plot_correlation(
        df_thermography,
        "aveOralM",
        "Results",
        "Correlation_Infrared_Thermography_Temperature.png",
    )


def fetch_Data():
    return preprocess_thermography_data("Data/InfraredThermographyTemperature.csv")


def perform_linear_regression(preprocessed_data):
    x_train, x_test, y_train, y_test = split_data(preprocessed_data, "aveOralM")
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    lr = LinearRegression()
    lr.fit(x_train_scaled, y_train)
    yh = lr.predict(x_test_scaled)

    plot_actual_vs_predicted(y_test, yh, "aveOralM", "Results")

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

    plot_residual(y_test, yh, "Results")
    plot_residual_distribution(y_test, yh, "Results")


def perform_MBS_linear_regression(preprocessed_data):
    x_train, x_test, y_train, y_test = split_data(preprocessed_data, "aveOralM")
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    lr = MiniBatchStochasticLinearRegression(batch_size=32)
    lr.fit(x_train_scaled, y_train)
    yh = lr.predict(x_test_scaled)

    plot_actual_vs_predicted(
        y_test, yh, "aveOralM", "Results", "miniBatchLinearRegression"
    )


def main():
    df = fetch_Data()

    plot_correlation_matrix(df, "Results", "Infrared_Thermography_Correlation_Matrix")

    plot_variance_inflation_factor(df, "Diabetes_binary", "Results")
    perform_linear_regression(df)

    # perform_MBS_linear_regression(df)
    perform_plotting(df)


if __name__ == "__main__":
    main()
