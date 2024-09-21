import pandas as pd

from Assignment1.Helpers import *

# Not much data to be pre-processed here.
df_diabetes = pd.read_csv("Data/DiabetesHealthIndicators.csv")

x_train, x_test, y_train, y_test = split_data(df_diabetes, "Diabetes_binary")
x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

plot_histogram(
    df_diabetes,
    "Results",
    "Histogram_Diabetes_Health_Indicators.png",
)
compute_and_plot_correlation(
    df_diabetes,
    "Diabetes_binary",
    "Results",
    "Correlation_Diabetes_Health_Indicators.png",
)
get_correlation(df_diabetes, "Diabetes_binary")
calculate_variance_inflation_factor(df_diabetes)

# Education vif = 29.72, correlation w/ target is low. Education and Income are correlated (0.44), could keep Income.
# AnyHealthcare vif = 20.93, correlation w/ target very low (0.016).
# ColCheck vif = 23.33, correlation w/ target low (0,064).
