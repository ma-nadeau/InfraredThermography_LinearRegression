# Regression Analysis Project

This project implements both linear regression and logistic regression from scratch. You can easily run the analysis on two different datasets by following the instructions below.

## How to Run

### Logistic Regression
To perform logistic regression on the **Diabetes Health Indicators** dataset:
1. Navigate to the `DiabetesHealthIndicators` folder.
2. Run the `PrepareAndTestDataset.py` file.

### Linear Regression
To perform linear regression on the **Infrared Thermography Temperature** dataset:
1. Navigate to the `InfraredThermographyTemperature` folder.
2. Run the `PrepareAndTestDataset.py` file.

## Datasets
The datasets are included in their respective folders:
- `DiabetesHealthIndicators/Data`: Contains the data for logistic regression.
- `InfraredThermographyTemperature/Data`: Contains the data for linear regression.

## Results
Any resulting figures/graphs will be saved in their respective folders:
- `DiabetesHealthIndicators/Results`: Contains the results for logistic regression.
- `InfraredThermographyTemperature/Results`: Contains the results for linear regression.

## Helper Functions
The `Helper.py` file contains all the helper functions used throughout this project. This file is shared between both regression models.

## Prerequisites
Make sure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
