import pandas as pd
from pandas import DataFrame


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



