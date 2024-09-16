import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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

    # Apply LabelEncoder on the Gender and Ethnicity categorical columns.
    df_thermography['Gender'] = label_encoder.fit_transform(df_thermography['Gender'])
    df_thermography['Ethnicity'] = label_encoder.fit_transform(df_thermography['Ethnicity'])

    # Delete rows with missing values.
    df_thermography = df_thermography.dropna()

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


preprocessed_data = preprocess_thermography_data('./InfraredThermographyTemperature.csv')
x_train, x_test, y_train, y_test = split_data(preprocessed_data)
x_train_scaled, x_test_scaled = scale_data(x_train, x_test)


