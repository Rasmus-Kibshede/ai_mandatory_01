import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def load_data(path: str):
    data = pd.read_csv(path)
    return data


def split(data: DataFrame):
    data[['make', 'model']] = data['car name'].str.split(n=1, expand=True)
    data.drop(columns=['car name'], inplace=True)
    data.to_csv('modified_dataset.csv', index=False)


def data_encoder(data: DataFrame, columns):
    pd.set_option('display.max_columns', None)
    data[columns[0]].unique()
    if len(columns) > 1:
        data[columns[1]].nunique()
    return pd.get_dummies(data, columns=columns)

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    return X_test, X_train, X_validation, y_test, y_train, y_validation



