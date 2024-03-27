import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame


def visualize_data(data: DataFrame):
    pairplot(data)
    describe_data(data)
    correlation_matrix(data.select_dtypes(include=['int64', 'float64']))
    boxplot(data)
    area_between_lines(data,
                       [('mpg', 'horsepower'), ('horsepower', 'weight'), ('displacement', 'mpg'),
                        ('acceleration', 'horsepower'), ('weight', 'displacement')])

    # Here we check scatter plots closer to see the outliers that where shown in the pairplot.
    scatterplot(data.select_dtypes(include=['int64', 'float64']),
                [('mpg', 'horsepower'), ('horsepower', 'weight'), ('mpg', 'displacement'),
                 ('acceleration', 'horsepower'), ('displacement', 'weight')])

    # Now we can analise the outliers in the dataset.


def pairplot(data: DataFrame):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sns.pairplot(data)
    plt.show()


def describe_data(data: DataFrame):
    print("Basic Statistics:")
    print('First few rows of DataFrame:\n ', data.head())
    print(data.describe())


def correlation_matrix(data: DataFrame):
    plt.figure(figsize=(10, 6))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()


def scatterplot(data: DataFrame, outliers):
    for pair in outliers:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=pair[0], y=pair[1], data=data)
        plt.title(f'Scatter Plot of {pair[0]} vs. {pair[1]}')
        plt.xlabel(pair[0])
        plt.ylabel(pair[1])
        plt.show()


def boxplot(data: DataFrame):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data.select_dtypes(include=['int64', 'float64']))
    plt.title('Box Plots of Numerical Columns')
    plt.xlabel('Columns')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()


def area_between_lines(data: DataFrame, outliers):
    for pair in outliers:
        make_avg_mpg = data.groupby('make')[pair[0]].mean()
        make_avg_horsepower = data.groupby('make')[pair[1]].mean()
        plt.figure(figsize=(10, 6))
        plt.plot(make_avg_mpg.index, make_avg_mpg.values, marker='o', linestyle='-', color='blue',
                 label=f'Average {pair[0]}')
        plt.bar(make_avg_horsepower.index, make_avg_horsepower.values, color='orange', label=f'Average {pair[1]}')
        plt.title(f'Average {pair[0]} and {pair[1]} by Car Make')
        plt.xlabel('Car Make')
        plt.ylabel(f'{pair[1]}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
