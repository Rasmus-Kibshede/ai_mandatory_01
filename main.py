from data import data_cleaner, data_handler, data_visualizer, regression_model
from data import regression_model, linear_regression_model

# Loads our modified data
data = data_handler.load_data('data/modified_dataset.csv')
# data_visualizer.visualize_data(data)

#regression_model.train_model(data)
linear_regression_model.train_model(data)


def prepare_data():
    original_data = data_handler.load_data('data/cars.csv')
    data_cleaner.clean(original_data)
