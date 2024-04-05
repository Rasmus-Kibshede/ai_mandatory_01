<<<<<<< HEAD
from data import data_cleaner, data_handler, data_visualizer
=======
from data import data_cleaner, data_handler
>>>>>>> fa6e828e74f011e2fde16d73f8c11ba5d0a15deb
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import numpy as np


data = data_handler.load_data('data/modified_dataset.csv')

#data_visualizer.visualize_data(data)

#TODO Run new models, untill sattisfied with accuracy

#TODO Load in functional models, test with validation data.

#TODO Divide into another file, this needs to be new models and load model.
print('\n|----------------------- Regression --------------------------|')
X = data.drop(['mpg'], axis=1)
y = data['mpg']

X = data_handler.data_encoder(X, ['make', 'model'])
X_test, X_train, X_validation, y_test, y_train, y_validation = data_handler.split_data(X, y)
print('| Training size:', len(X_train), '| Testing size:', len(X_test), '| Validation size:', len(X_validation), '|\n')

models = [MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000),
          LinearRegression(),
          Ridge(alpha=1.0),
          Lasso(alpha=1.0),
          DecisionTreeRegressor(max_depth=5),
          GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
          RandomForestRegressor(n_estimators=100, random_state=42)]

best_model_name = ""
best_mse = float('inf')
best_rmse = float('inf')
best_r_squared = -float('inf')

index = 0
for model in models:
    index += 1
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{model.__class__.__name__} MAE:", mae)
    # Skal denne med i stedet for MAE?
    # print(f"{model.__class__.__name__} MSE:", mse)
    print(f"{model.__class__.__name__} RMSE:", rmse)
    print(f"{model.__class__.__name__} R-squared:", r2, '\n')
    if mse < best_mse and rmse < best_rmse and r2 > best_r_squared:
        best_mse = mse
        best_rmse = rmse
        best_r_squared = r2
        best_model_name = model.__class__.__name__
        best_model = model

joblib.dump(model, f"{best_model_name}.pkl")

loaded_model = joblib.load(f"{best_model_name}.pkl")
predictions = loaded_model.predict(X_validation)
mae = mean_absolute_error(y_validation, predictions)
mse = mean_squared_error(y_validation, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_validation, predictions)
print(f'Validation data evaluation of {best_model_name}: \n')
print(f"{model.__class__.__name__} MAE:", mae)
print(f"{model.__class__.__name__} MSE:", mse)
print(f"{model.__class__.__name__} RMSE:", rmse)
print(f"{model.__class__.__name__} R-squared:", r2, '\n')

plt.figure(figsize=(10, 6))
plt.scatter(y_validation, predictions)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title(f'BIS R-Model: {best_model_name}')
plt.plot([y_validation.min(), y_validation.max()], [y_validation.min(), y_validation.max()], 'k--', lw=4)
plt.show()

#TODO Divide into another file, this needs to be new models and load model.
print('\n|-------------------- Classification -------------------------|')
X = data.drop(columns=['make'])
X_transformed = data_handler.data_encoder(X, ['model'])
y = data['make']
X_test, X_train, X_validation, y_test, y_train, y_validation = data_handler.split_data(X_transformed, y)
print('| Training size:', len(X_train), '| Testing size:', len(X_test), '| Validation size:', len(X_validation), '|\n')

models = [KNeighborsClassifier(n_neighbors=4),
          MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10000, random_state=42, learning_rate='adaptive',
                        alpha=0.0001),
          DecisionTreeClassifier(criterion='gini', max_depth=14, min_samples_split=2, min_samples_leaf=1),
          RandomForestClassifier(n_estimators=8, random_state=42),
          GradientBoostingClassifier(max_depth=3000, learning_rate=0.001, random_state=42),
          GaussianNB()]

best_model_name = ""
best_accuracy = 0.0

for model in models:
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_train_pred)
    print(f"{model.__class__.__name__}: Accuracy Training data:", accuracy)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{model.__class__.__name__}: Accuracy:", accuracy, '\n')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model.__class__.__name__

joblib.dump(model, f"{best_model_name}.pkl")

loaded_model = joblib.load(f"{best_model_name}.pkl")
predictions = loaded_model.predict(X_validation)
accuracy = accuracy_score(y_validation, predictions)

print(f'Validation data evaluation of {best_model_name}:')
print(f'Accuracy on validation data: {best_accuracy}')


def prepare_data():
    original_data = data_handler.load_data('data/cars.csv')
    data_cleaner.clean(original_data)
