from LogisticRegression import *
from MiniBatchLogisticRegression import *
from Assignment1.Helpers import *
from Assignment1.PlotHelpers import *

# from minibatchLogR import *


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
        perform_logistic_regression(df, model_type="mini_batch")
    else:
        perform_logistic_regression(df)


def perform_logistic_regression(
    df, target_variable="Diabetes_binary", model_type="standard"
):
    x_train, x_test, y_train, y_test = undersampling_dataset(df, target_variable)

    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    if model_type == "mini_batch":
        lr = MiniBatchLogisticRegression()
        model_name = "Mini_Batch_Logistic_Regression"
    else:
        lr = LogisticRegression()
        model_name = "Logistic_Regression"

    lr.fit(x_train_scaled, y_train, optimization=True)
    yh_bool, yh_real = lr.predict(x_test_scaled)

    plot_residual(y_test, yh_bool, "Results", f"{model_name}_Residual.png")
    plot_residual_distribution(
        y_test, yh_bool, "Results", f"{model_name}_Residual_Distribution.png"
    )

    # Plot confusion matrix.
    plot_confusion_matrix(y_test, yh_bool, title=f"{model_name}_Confusion_Matrix.png")

    # Plot statistics.
    print_logistic_regression_model_stats(x_test_scaled, y_test, yh_bool)

    # Plot ROC curve.
    plot_roc_curve(y_test, yh_real)


def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    """Helper function to fit the model and calculate metrics."""
    # Fit the model
    model.fit(X_train, Y_train)

    # Make predictions
    y_pred_train, _ = model.predict(X_train)
    y_pred_test, _ = model.predict(X_test)

    # Compute MSE and R² for both train and test
    train_accuracy = accuracy_score(Y_train, y_pred_train)
    test_accuracy = accuracy_score(Y_test, y_pred_test)
    train_f1 = f1_score(Y_train, y_pred_train)
    test_f1 = f1_score(Y_test, y_pred_test)
    train_precision = precision_score(Y_train, y_pred_train)
    test_precision = precision_score(Y_test, y_pred_test)
    train_recall = recall_score(Y_train, y_pred_train)
    test_recall = recall_score(Y_test, y_pred_test)

    return (
        train_accuracy,
        test_accuracy,
        train_f1,
        test_f1,
        train_precision,
        test_precision,
        train_recall,
        test_recall,
    )


def logistic_regression_test_growing_subset(df, target_variable="Diabetes_binary"):
    x_train, x_test, y_train, y_test = undersampling_dataset(df, target_variable)

    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    results = {
        "log_acc_train": [],
        "log_acc_test": [],
        "log_f1_train": [],
        "log_f1_test": [],
        "log_pre_train": [],
        "log_pre_test": [],
        "log_rec_train": [],
        "log_rec_test": [],
        "mini_acc_train": [],
        "mini_acc_test": [],
        "mini_f1_train": [],
        "mini_f1_test": [],
        "mini_pre_train": [],
        "mini_pre_test": [],
        "mini_rec_train": [],
        "mini_rec_test": [],
    }
    train_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for size in train_sizes:
        indices = np.random.choice(
            x_train_scaled.shape[0],
            size=int(size * x_train_scaled.shape[0]),
            replace=False,
        )
        x_train_subset = x_train_scaled[indices]
        y_train_subset = y_train.iloc[indices]

        Logistic_gd = LogisticRegression()
        (
            log_acc_train,
            log_acc_test,
            log_f1_train,
            log_f1_test,
            log_pre_train,
            log_pre_test,
            log_rec_train,
            log_rec_test,
        ) = evaluate_model(
            Logistic_gd, x_train_subset, y_train_subset, x_test_scaled, y_test
        )

        results["log_acc_train"].append(log_acc_train)
        results["log_acc_test"].append(log_acc_test)
        results["log_f1_train"].append(log_f1_train)
        results["log_f1_test"].append(log_f1_test)
        results["log_pre_train"].append(log_pre_train)
        results["log_pre_test"].append(log_pre_test)
        results["log_rec_train"].append(log_rec_train)
        results["log_rec_test"].append(log_rec_test)

        minibatch = MiniBatchLogisticRegression()
        (
            mini_acc_train,
            mini_acc_test,
            mini_f1_train,
            mini_f1_test,
            mini_pre_train,
            mini_pre_test,
            mini_rec_train,
            mini_rec_test,
        ) = evaluate_model(
            minibatch, x_train_subset, y_train_subset, x_test_scaled, y_test
        )

        results["mini_acc_train"].append(mini_acc_train)
        results["mini_acc_test"].append(mini_acc_test)
        results["mini_f1_train"].append(mini_f1_train)
        results["mini_f1_test"].append(mini_f1_test)
        results["mini_pre_train"].append(mini_pre_train)
        results["mini_pre_test"].append(mini_pre_test)
        results["mini_rec_train"].append(mini_rec_train)
        results["mini_rec_test"].append(mini_rec_test)

    plot_acc_growing_set(results, train_sizes)
    plot_f1_growing_set(results, train_sizes)
    plot_pre_growing_set(results, train_sizes)
    plot_rec_growing_set(results, train_sizes)


def mbsgd_test_batch_sizes(df, target_variable="Diabetes_binary"):
    x_train, x_test, y_train, y_test = split_data(
        df, target_variable, test_size=0.2, random_state=42
    )
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    batch_sizes = [8, 16, 32, 64, 128]
    ''', x_train_scaled.shape[0]'''

    # Dictionary to store losses for different batch sizes
    losses_by_batch_size = {}
    r2s_by_batch_size = {}

    # Train models with different batch sizes and record training loss over iterations
    for batch_size in batch_sizes:
        model = MiniBatchLogisticRegression(batch_size=batch_size, learning_rate=0.003)
        _, losses = model.fit(x_train_scaled, y_train)
        losses_by_batch_size[batch_size] = losses
    plot_batch_sizes_result(losses_by_batch_size, {}, 1, title1="Task3.4_logistic")


def logistic_regression_test_learning_rates(df, target_variable="Diabetes_binary"):
    x_train, x_test, y_train, y_test = undersampling_dataset(df, target_variable)

    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    results = {
        "log_acc_train": [],
        "log_acc_test": [],
        "log_f1_train": [],
        "log_f1_test": [],
        "log_pre_train": [],
        "log_pre_test": [],
        "log_rec_train": [],
        "log_rec_test": [],
        "mini_acc_train": [],
        "mini_acc_test": [],
        "mini_f1_train": [],
        "mini_f1_test": [],
        "mini_pre_train": [],
        "mini_pre_test": [],
        "mini_rec_train": [],
        "mini_rec_test": [],
    }
    learning_rates = [0.01, 0.02, 0.05]
    for lr in learning_rates:
        Logistic_gd = LogisticRegression(learning_rate=lr)

        (
            log_acc_train,
            log_acc_test,
            log_f1_train,
            log_f1_test,
            log_pre_train,
            log_pre_test,
            log_rec_train,
            log_rec_test,
        ) = evaluate_model(Logistic_gd, x_train_scaled, y_train, x_test_scaled, y_test)

        results["log_acc_train"].append(log_acc_train)
        results["log_acc_test"].append(log_acc_test)
        results["log_f1_train"].append(log_f1_train)
        results["log_f1_test"].append(log_f1_test)
        results["log_pre_train"].append(log_pre_train)
        results["log_pre_test"].append(log_pre_test)
        results["log_rec_train"].append(log_rec_train)
        results["log_rec_test"].append(log_rec_test)

        minibatch = MiniBatchLogisticRegression(learning_rate=lr)
        (
            mini_acc_train,
            mini_acc_test,
            mini_f1_train,
            mini_f1_test,
            mini_pre_train,
            mini_pre_test,
            mini_rec_train,
            mini_rec_test,
        ) = evaluate_model(minibatch, x_train_scaled, y_train, x_test_scaled, y_test)

        results["mini_acc_train"].append(mini_acc_train)
        results["mini_acc_test"].append(mini_acc_test)
        results["mini_f1_train"].append(mini_f1_train)
        results["mini_f1_test"].append(mini_f1_test)
        results["mini_pre_train"].append(mini_pre_train)
        results["mini_pre_test"].append(mini_pre_test)
        results["mini_rec_train"].append(mini_rec_train)
        results["mini_rec_test"].append(mini_rec_test)

    plot_logistic_learning_rates(results, learning_rates)


def main():
    df = preprocess_diabetes_data("Data/DiabetesHealthIndicators.csv")

    # Perform regression testing for standard and mini-batch.
    # run_regression_tests(df, "Test logistic regression", regression_type="standard")
    # run_regression_tests(
    #     df, "Test mini-batch logistic regression", regression_type="mini_batch"
    # )

    # logistic_regression_test_growing_subset(df)
    #
    # mbsgd_test_batch_sizes(df)
    logistic_regression_test_learning_rates(df)
    plot_histogram(df, "Results", "Histogram_Diabetes_Health_Indicators.png")
    plot_correlation_matrix(df, "Results", "CDC_Correlation_Matrix")

    plot_variance_inflation_factor(df, "Diabetes_binary", "Results")


if __name__ == "__main__":
    main()
