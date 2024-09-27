from LogisticRegression import *
from MiniBatchLogisticRegression import *
from Assignment1.Helpers import *


def preprocess_diabetes_data(file_name):
    # Load the thermography dataset into a Pandas dataframe.
    df_diabetes = pd.read_csv(file_name)

    # Remove ID column.
    df_diabetes.drop(columns=["ID"], inplace=True)

    # Remove duplicates.
    df_diabetes.drop_duplicates(inplace=True)

    # Remove rows with empty values.
    # Explain choice in report.
    df_diabetes.dropna(inplace=True)

    # Dropping these columns does not have a significant effect on outcome.
    # Keep for report.
    df_diabetes.drop(
        columns=["Fruits", "Veggies", "AnyHealthcare", "NoDocbcCost", "Sex"],
        inplace=True,
    )
    return df_diabetes


def run_regression_tests(df, label, regression_type="standard"):
    print(f"\n{label}:")

    if regression_type == "mini_batch":
        perform_logistic_regression(df, model_type="mini_batch", batch_size=32)
    else:
        perform_logistic_regression(df)


def perform_logistic_regression(df, target_variable="Diabetes_binary", model_type="standard", batch_size=None):
    x_train, x_test, y_train, y_test = oversampling_dataset(
        df, target_variable
    )

    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    if model_type == "mini_batch":
        lr = MiniBatchLogisticRegression(batch_size)
        model_name = "Mini_Batch_Logistic_Regression"
    else:
        lr = LogisticRegression()
        model_name = "Logistic_Regression"

    lr.fit(x_train_scaled, y_train, optimization=True)
    yh_bool, yh_real = lr.predict(x_test_scaled)

    plot_residual(y_test, yh_bool, "Results", f"{model_name}_Residual.png")
    plot_residual_distribution(y_test, yh_bool, "Results", f"{model_name}_Residual_Distribution.png")

    # Plot confusion matrix.
    plot_confusion_matrix(y_test, yh_bool, title=f"{model_name}_Confusion_Matrix.png")

    # Plot statistics.
    print_logistic_regression_model_stats(x_test_scaled, y_test, yh_bool)

    # Plot ROC curve.
    plot_roc_curve(y_test, yh_real)


def main():
    df = preprocess_diabetes_data("Data/DiabetesHealthIndicators.csv")

    # Perform regression testing for standard and mini-batch.
    run_regression_tests(df, "Test logistic regression", regression_type="standard")
    run_regression_tests(df, "Test mini-batch logistic regression", regression_type="mini_batch")

    plot_histogram(
        df, "Results", "Histogram_Diabetes_Health_Indicators.png"
    )
    plot_correlation_matrix(df, "Results", "CDC_Correlation_Matrix")

    plot_variance_inflation_factor(df, "Diabetes_binary", "Results")

    perform_logistic_regression(df)


if __name__ == "__main__":
    main()
