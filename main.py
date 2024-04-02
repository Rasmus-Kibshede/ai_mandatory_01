from data import data_cleaner, data_handler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn import preprocessing
import joblib
import numpy as np

data = data_handler.load_data('data/modified_dataset.csv')

print('\n|----------------------- Regression --------------------------|')
X = data.drop(['mpg'], axis=1)
y = data['mpg']
lab = preprocessing.LabelEncoder()
y_transform = lab.fit_transform(y)

X = data_handler.data_encoder(X, ['make', 'model'])
X_test, X_train, X_validation, y_test, y_train, y_validation = data_handler.split_data(X, y_transform)
print('| Training size:', len(X_train), '| Testing size:', len(X_test), '| Validation size:', len(X_validation), '|\n')

models = [MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000),
          LinearRegression(),
          Ridge(alpha=1.0),
          Lasso(alpha=1.0),
          DecisionTreeRegressor(max_depth=5),
          RandomForestRegressor(n_estimators=100, max_depth=5),
          GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)]

best_model_name = ""
best_mse = float('inf')  # Initialize with infinity for finding minimum MSE
best_rmse = float('inf')  # Initialize with infinity for finding minimum RMSE
best_r_squared = -float('inf')  # Initialize with negative infinity for finding maximum R-squared

index = 0
for model in models:
    index += 1
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{model.__class__.__name__} MSE:", mse)
    print(f"{model.__class__.__name__} RMSE:", rmse)
    print(f"{model.__class__.__name__} R-squared:", r2, '\n')
    if mse < best_mse and rmse < best_rmse and r2 > best_r_squared:
        best_mse = mse
        best_rmse = rmse
        best_r_squared = r2
        best_model_name = model.__class__.__name__
        best_model = model  # Store the best model object

joblib.dump(model, f"{best_model_name}.pkl")

loaded_model = joblib.load(f"{best_model_name}.pkl")
predictions = loaded_model.predict(X_validation)
mse = mean_squared_error(y_validation, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_validation, predictions)
print(f'Validation data evaluation of {best_model_name}: \n')
print(f"{model.__class__.__name__} MSE:", mse)
print(f"{model.__class__.__name__} RMSE:", rmse)
print(f"{model.__class__.__name__} R-squared:", r2, '\n')

print('\n|-------------------- Classification -------------------------|')
X = data.drop(columns=['make', 'model'])
y = data['make']
y = lab.fit_transform(y)
X_test, X_train, X_validation, y_test, y_train, y_validation = data_handler.split_data(X, y)
print('| Training size:', len(X_train), '| Testing size:', len(X_test), '| Validation size:', len(X_validation), '|\n')

models = [KNeighborsClassifier(n_neighbors=4),  # 3 and 4 looks like the most optimized
          MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42, learning_rate='adaptive'),
          DecisionTreeClassifier(criterion='gini', max_depth=50, min_samples_split=2, min_samples_leaf=1),
          RandomForestClassifier(n_estimators=400, random_state=42),
          GradientBoostingClassifier(max_depth=50, learning_rate=0.001, random_state=42),
          GaussianNB()]

best_model_name = ""
best_accuracy = 0.0

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{model.__class__.__name__}: Accuracy:", accuracy, '\n')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model.__class__.__name__

joblib.dump(model, f"{best_model_name}.pkl")

loaded_model = joblib.load(f"{best_model_name}.pkl")
predictions = loaded_model.predict(X_validation)
predictions = lab.inverse_transform(predictions)
y_validation = lab.inverse_transform(y_validation)

# [print("Prediction:", predictions[i], "\nActual Data:", y_validation[i], '\n') for i in
#    range(len(predictions))]
accuracy = accuracy_score(y_validation, predictions)

print(f'Validation data evaluation of {best_model_name}: \n')
print('Accuracy on validation data:', accuracy, '\n')


# Result is always 9/40 and its the same every time, something is wrong when transforming the data.
# correct_count = sum([1 for i in range(len(predictions)) if predictions[i] == y_validation[i]])

# [print("Prediction:", predictions[i], "\nActual Data:", y_validation[i], "\nPrediction is correct!\n" if predictions[i] == y_validation[i] else "\nPrediction is incorrect!\n") for i in range(len(predictions))]

# print("Total correct predictions:", correct_count, "/", len(predictions))
# print('Percentage:', (correct_count/len(predictions)) * 100, '%')


def prepare_data():
    original_data = data_handler.load_data('data/cars.csv')
    data_cleaner.clean(original_data)
