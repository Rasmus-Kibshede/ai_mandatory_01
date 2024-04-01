from data import data_cleaner, data_handler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

data = data_handler.load_data('data/modified_dataset.csv')

print('\n|----------------------- Regression --------------------------|')
X = data.drop(['mpg'], axis=1)
y = data['mpg']
lab = preprocessing.LabelEncoder()
y = lab.fit_transform(y)
X = data_handler.data_encoder(X, ['make', 'model'])
X_test, X_train, X_validation, y_test, y_train, y_validation = data_handler.split_data(X, y)
print('| Training size:', len(X_train), '| Testing size:', len(X_test), '| Validation size:', len(X_validation), '|\n')

# regression_model.train_model(data)
# linear_regression_model.train_model(data)


models = [MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000),
          LinearRegression(),
          Ridge(alpha=1.0),
          Lasso(alpha=1.0),
          DecisionTreeRegressor(max_depth=5),
          RandomForestRegressor(n_estimators=100, max_depth=5),
          GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)]

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{model.__class__.__name__} MSE:", mse)
    print(f"{model.__class__.__name__} RMSE:", rmse)
    print(f"{model.__class__.__name__} R-squared:", r2)


print('\n|-------------------- Classification -------------------------|')
X = data.drop(columns=['make', 'model'])
y = data['make']
y = lab.fit_transform(y)
X_test, X_train, X_validation, y_test, y_train, y_validation = data_handler.split_data(X, y)
print('| Training size:', len(X_train), '| Testing size:', len(X_test), '| Validation size:', len(X_validation), '|\n')

models = [KNeighborsClassifier(n_neighbors=4),
          MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
          DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1),
          RandomForestClassifier(n_estimators=100, random_state=42),
          GradientBoostingClassifier(),
          GaussianNB()]

for model in models:
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__}: Accuracy:", accuracy)


def prepare_data():
    original_data = data_handler.load_data('data/cars.csv')
    data_cleaner.clean(original_data)
