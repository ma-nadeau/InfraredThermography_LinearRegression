import pandas as pd
import numpy as np
from LogisticRegression import *
from Assignment1.Helpers import *

# Not much data to be pre-processed here.


def fetch_data():
    return pd.read_csv("Data/DiabetesHealthIndicators.csv")


def perform_logistic_regression(preprocessed_data):

    preprocessed_data = preprocessed_data.dropna()
    preprocessed_data = preprocessed_data.drop(columns=["ID"])

    x_train, x_test, y_train, y_test = split_data(preprocessed_data, "Diabetes_binary")
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    lr = LogisticRegression(0.05, 1000, 1e-8, True)
    lr.fit(x_train_scaled, y_train)
    yh_bool, yh_real = lr.predict(x_test_scaled)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, yh_bool)

    # Plot ROC curve
    plot_roc_curve(y_test, yh_real)


def main():
    df = fetch_data()
    # plot_histogram(df,"Results", "Histogram_Diabetes_Health_Indicators.png")
    # compute_and_plot_correlation(df, "Diabetes_binary", "Results", "Correlation_Diabetes_Health_Indicators.png")
    # calculate_variance_inflation_factor(df)
    preprocessed_data = df.dropna()
    preprocessed_data = preprocessed_data.drop(columns=["ID"])
    plot_histogram(
        preprocessed_data, "Results", "Histogram_Diabetes_Health_Indicators.png"
    )
    plot_correlation_matrix(preprocessed_data, "Results", "CDC_Correlation_Matrix")

    plot_variance_inflation_factor(preprocessed_data, "Diabetes_binary", "Results")

    perform_logistic_regression(df)


if __name__ == "__main__":
    main()

# Education vif = 29.72, correlation w/ target is low. Education and Income are correlated (0.44), could keep Income.
# AnyHealthcare vif = 20.93, correlation w/ target very low (0.016).
# ColCheck vif = 23.33, correlation w/ target low (0,064).
