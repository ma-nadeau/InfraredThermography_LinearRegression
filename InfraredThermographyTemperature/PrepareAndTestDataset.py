from sklearn.preprocessing import LabelEncoder
from InfraredThermographyTemperature.Helpers import *
from InfraredThermographyTemperature.PlotHelpers import *
from LinearRegression import LinearRegression
from MiniBatchStochasticLinearRegression import *


def preprocess_thermography_data(file_name):
    """
    Preprocesses the thermography data by loading, cleaning, encoding, and handling outliers.

    Parameters:
    file_name (str): The file path of the thermography data CSV file.

    Returns:
    pd.DataFrame: Preprocessed dataframe ready for analysis.
    """
    # Load the thermography dataset into a Pandas dataframe.
    df_thermography = pd.read_csv(file_name)

    # Initialize encoder for categorical columns.
    label_encoder = LabelEncoder()

    # Remove SubjectID column.
    df_thermography.drop("SubjectID", axis=1, inplace=True)

    # Remove duplicates from the dataframe.
    df_thermography.drop_duplicates(inplace=True)

    # Function to calculate midpoint for age ranges.
    def get_midpoint(age_range):
        """Converts an age range into a midpoint value."""
        if "-" in age_range:
            lower, upper = age_range.split("-")
            return (int(lower) + int(upper)) / 2
        else:
            lower = int(age_range[1:])  # Extract the number after ">"
            return lower + 5  # Estimate for population over 60 years.

    # Apply the midpoint function to the 'Age' column.
    df_thermography["Age"] = df_thermography["Age"].apply(get_midpoint)

    # Function to remove outliers based on Interquartile Range (IQR).
    def remove_outliers(df, column_name):
        """Removes outliers in a specific column using IQR method."""
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 2.5 * iqr
        upper_bound = q3 + 2.5 * iqr
        df.drop(df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)].index, inplace=True)

    # Remove outliers for 'Age', 'Distance', and 'Humidity' columns.
    remove_outliers(df_thermography, "Age")
    remove_outliers(df_thermography, "Distance")
    remove_outliers(df_thermography, "Humidity")

    # Apply LabelEncoder to the 'Gender' and 'Ethnicity' columns.
    df_thermography["Gender"] = label_encoder.fit_transform(df_thermography["Gender"])
    df_thermography["Ethnicity"] = label_encoder.fit_transform(df_thermography["Ethnicity"])

    # Replace missing values with column mean.
    df_thermography = df_thermography.fillna(df_thermography.mean())

    return df_thermography


def plot_histogram_correlation(df):
    """
    Plots histograms and correlation matrix for the thermography dataset.

    Parameters:
    df (pd.DataFrame): The thermography dataframe.
    """
    # Plot histogram of the dataframe.
    plot_histogram(df, "Results", "Histogram_Infrared_Thermography_Temperature.png")

    # Compute and plot the correlation matrix for the target column 'aveOralM'.
    compute_and_plot_correlation(df, "aveOralM", "Results", "Correlation_Infrared_Thermography_Temperature.png")


def perform_linear_regression(df, target_variable="aveOralM", model_type="standard", batch_size=None):
    """
    Performs linear regression (either standard or mini-batch) on the dataset.

    Parameters:
    df (pd.DataFrame): The thermography dataframe.
    target_variable (str, optional): The target variable for regression.
    model_type (str, optional): Type of regression model ('standard' or 'mini_batch').
    batch_size (int, optional): Batch size for mini-batch regression.
    """
    # Split the data before scaling.
    x_train, x_test, y_train, y_test = split_data(df, target_variable, test_size=0.3, random_state=42)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    # Initialize the appropriate regression model.
    if model_type == "mini_batch":
        lr = MiniBatchStochasticLinearRegression(batch_size=batch_size)
        model_name = "Mini_Batch_Linear_Regression"
    else:
        lr = LinearRegression()
        model_name = "Linear_Regression"

    # Fit the model and make predictions.
    lr.fit(x_train_scaled, y_train, optimization=False)
    yh = lr.predict(x_test_scaled)

    # Plot and print model evaluation metrics.
    plot_actual_vs_predicted(y_test, yh, target_variable, "Results", model_name)
    print_linear_regression_model_stats(x_test, y_test, yh)
    plot_residual(y_test, yh, "Results", f"{model_name}_Residual.png")
    plot_residual_distribution(y_test, yh, "Results", f"{model_name}_Residual_Distribution.png")


def run_regression_tests(df, label, regression_type="standard"):
    """
    Runs regression tests (standard or mini-batch) on the dataset.

    Parameters:
    df (pd.DataFrame): The thermography dataframe.
    label (str): Label for the test being performed.
    regression_type (str, optional): Type of regression ('standard' or 'mini_batch').
    """
    print(f"\n{label}:")

    # Call perform_linear_regression with the appropriate model type.
    if regression_type == "mini_batch":
        perform_linear_regression(df, model_type="mini_batch", batch_size=32)
    else:
        perform_linear_regression(df)


def linear_regression_test_growing_subset(df, target_variable="aveOralM"):
    """
    Tests linear regression performance as the training subset size increases.

    Parameters:
    df (pd.DataFrame): The thermography dataframe.
    target_variable (str, optional): The target variable for regression.
    """
    def evaluate_model(model, X_train, Y_train, X_test, Y_test):
        """Helper function to fit the model and calculate performance metrics."""
        model.fit(X_train, Y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_mse = mean_squared_error(Y_train, y_pred_train)
        test_mse = mean_squared_error(Y_test, y_pred_test)
        train_r2 = r2_score(Y_train, y_pred_train)
        test_r2 = r2_score(Y_test, y_pred_test)
        train_evs = explained_variance_score(Y_train, y_pred_train)
        test_evs = explained_variance_score(Y_test, y_pred_test)
        return train_mse, test_mse, train_r2, test_r2, train_evs, test_evs

    # Split and scale the data.
    x_train, x_test, y_train, y_test = split_data(df, target_variable, test_size=0.2, random_state=42)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    # Dictionary to store results.
    results = {
        "lin_mse_train": [], "lin_mse_test": [], "lin_r2_train": [], "lin_r2_test": [],
        "lin_evs_train": [], "lin_evs_test": [], "mini_mse_train": [], "mini_mse_test": [],
        "mini_r2_train": [], "mini_r2_test": [], "mini_evs_train": [], "mini_evs_test": []
    }

    # Test for different training subset sizes.
    train_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for size in train_sizes:
        indices = np.random.choice(x_train_scaled.shape[0], size=int(size * x_train_scaled.shape[0]), replace=False)
        x_train_subset = x_train_scaled[indices]
        y_train_subset = y_train.iloc[indices]

        # Evaluate the analytical linear regression model.
        analytical = LinearRegression()
        lin_mse_train, lin_mse_test, lin_r2_train, lin_r2_test, lin_evs_train, lin_evs_test = evaluate_model(
            analytical, x_train_subset, y_train_subset, x_test_scaled, y_test)

        # Store results.
        results["lin_mse_train"].append(lin_mse_train)
        results["lin_mse_test"].append(lin_mse_test)
        results["lin_r2_train"].append(lin_r2_train)
        results["lin_r2_test"].append(lin_r2_test)
        results["lin_evs_train"].append(lin_evs_train)
        results["lin_evs_test"].append(lin_evs_test)

        # Evaluate the mini-batch stochastic linear regression model.
        minibatch = MiniBatchStochasticLinearRegression()
        mini_mse_train, mini_mse_test, mini_r2_train, mini_r2_test, mini_evs_train, mini_evs_test = evaluate_model(
            minibatch, x_train_subset, y_train_subset, x_test_scaled, y_test)

        # Store mini-batch results.
        results["mini_mse_train"].append(mini_mse_train)
        results["mini_mse_test"].append(mini_mse_test)
        results["mini_r2_train"].append(mini_r2_train)
        results["mini_r2_test"].append(mini_r2_test)
        results["mini_evs_train"].append(mini_evs_train)
        results["mini_evs_test"].append(mini_evs_test)

    # Plot the results for MSE, R², and EVS.
    plot_mse_growing_set(results, train_sizes)
    plot_r2_growing_set(results, train_sizes)
    plot_evs_growing_set(results, train_sizes)


def mbsgd_test_batch_sizes(df, target_variable="aveOralM"):
    """
    Tests Mini-Batch Stochastic Gradient Descent (MBSGD) with different batch sizes.

    Parameters:
    df (pd.DataFrame): The thermography dataframe.
    target_variable (str, optional): The target variable for regression.
    """
    x_train, x_test, y_train, y_test = split_data(df, target_variable, test_size=0.2, random_state=42)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    batch_sizes = [8, 16, 32, 64, 128, x_train_scaled.shape[0]]

    # Dictionary to store losses and R² scores by batch size.
    losses_by_batch_size = {}
    r2s_by_batch_size = {}

    # Train models with different batch sizes and record results.
    for batch_size in batch_sizes:
        model = MiniBatchStochasticLinearRegression(learning_rate=0.1, batch_size=batch_size, epoch=100)
        losses, r2s = model.fit(x_train_scaled, y_train)
        losses_by_batch_size[batch_size] = losses
        r2s_by_batch_size[batch_size] = r2s

    # Plot the results.
    plot_batch_sizes_result(losses_by_batch_size, r2s_by_batch_size)


def linear_regression_learning_rate_test(df, target_variable="aveOralM"):
    """
    Tests Mini-Batch Stochastic Gradient Descent (MBSGD) with different learning rates.

    Parameters:
    df (pd.DataFrame): The thermography dataframe.
    target_variable (str, optional): The target variable for regression.
    """
    def evaluate_model(model, X_train, Y_train, X_test, Y_test):
        """Helper function to fit the model and calculate metrics."""
        model.fit(X_train, Y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_mse = mean_squared_error(Y_train, y_pred_train)
        test_mse = mean_squared_error(Y_test, y_pred_test)
        train_r2 = r2_score(Y_train, y_pred_train)
        test_r2 = r2_score(Y_test, y_pred_test)
        return train_mse, test_mse, train_r2, test_r2

    x_train, x_test, y_train, y_test = split_data(df, target_variable, test_size=0.2, random_state=42)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    # Dictionary to store results by learning rate.
    results = {
        "mini_mse_train": [], "mini_mse_test": [], "mini_r2_train": [], "mini_r2_test": []
    }
    learning_rates = [0.005, 0.01, 0.5]

    # Test for different learning rates.
    for lr in learning_rates:
        minibatch = MiniBatchStochasticLinearRegression(learning_rate=lr)
        mini_mse_train, mini_mse_test, mini_r2_train, mini_r2_test = evaluate_model(
            minibatch, x_train_scaled, y_train, x_test_scaled, y_test)

        # Store results.
        results["mini_mse_train"].append(mini_mse_train)
        results["mini_mse_test"].append(mini_mse_test)
        results["mini_r2_train"].append(mini_r2_train)
        results["mini_r2_test"].append(mini_r2_test)

    # Plot the learning rate results.
    plot_linear_learning_rates(results, learning_rates)


def main():
    """
    Main function to run preprocessing, regression tests, and analysis on thermography dataset.
    """
    # Preprocess the dataset.
    df = preprocess_thermography_data("Data/InfraredThermographyTemperature.csv")

    # Perform regression tests before dropping columns.
    run_regression_tests(df, "Test without dropping columns, analytical regression", regression_type="standard")
    run_regression_tests(df, "Test without dropping columns, mini-batch regression", regression_type="mini_batch")

    # Plot the correlation matrix before dropping columns.
    plot_correlation_matrix(df, "Results", "Infrared_Thermography_Correlation_Matrix_Before_Dropping_Values.png")

    # Drop columns with high correlation.
    columns_to_drop = [
        "T_RC_Dry1", "T_RC_Wet1", "T_RC_Max1", "T_LC_Dry1", "T_LC_Wet1", "T_LC_Max1",
        "canthi4Max1", "T_FHRC1", "T_FHLC1", "T_FHBC1", "T_FHTC1", "T_OR1", "T_Max1",
        "T_FH_Max1", "T_FHC_Max1", "LCC1", "canthiMax1", "Max1R13_1", "aveAllL13_1",
        "T_RC1", "T_LC1", "RCC1", "aveOralF", "Gender", "Age", "Ethnicity", "T_atm",
        "Humidity", "Distance"
    ]
    df.drop(columns_to_drop, axis=1, inplace=True)

    # Perform regression tests after dropping columns.
    run_regression_tests(df, "Test after dropping columns, analytical regression", regression_type="standard")
    run_regression_tests(df, "Test after dropping columns, mini-batch regression", regression_type="mini_batch")

    # Plot the correlation matrix after dropping columns.
    plot_correlation_matrix(df, "Results", "Infrared_Thermography_Correlation_Matrix_After_Dropping_Values.png")

    # Test linear regression with growing subset sizes.
    linear_regression_test_growing_subset(df)

    # Plot histogram and correlation after processing.
    plot_histogram_correlation(df)

    # Test mini-batch SGD with varying batch sizes.
    mbsgd_test_batch_sizes(df)

    # Test mini-batch SGD with different learning rates.
    linear_regression_learning_rate_test(df)

    # Plot variance inflation factor (VIF).
    plot_variance_inflation_factor(df, "Diabetes_binary", "Results")


if __name__ == "__main__":
    main()
