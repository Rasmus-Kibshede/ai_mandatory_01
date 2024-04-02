from keras import Sequential
from keras.models import load_model
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

from data import data_handler


def train_model(data):
    ##TODO NAVNGIVNING SKAL LAVES OM!
    ##TODO Skal have en save model metode
    ##DATA SPLIT metode
    ##Model load metode osv.

    X_regression = data.drop(['mpg'], axis=1)
    y_regression = data['mpg']
    X_regression = data_handler.data_encoder(X_regression, ['make', 'model'])

    # Splits original dataset into 70/30
    X_regression_train, X_temp, y_regression_train, y_temp = train_test_split(X_regression, y_regression,
                                                                              test_size=0.3, random_state=42)
    # Splits test set to 50/50
    X_regression_test, X_prediction, y_regression_test, y_prediction = train_test_split(X_temp, y_temp,
                                                                                        test_size=0.5, random_state=42)

    X_regression_train = X_regression_train.astype(np.float32)
    y_regression_train = y_regression_train.astype(np.float32)
    X_regression_test = X_regression_test.astype(np.float32)
    y_regression_test = y_regression_test.astype(np.float32)
    X_prediction = X_prediction.astype(np.float32)

    # Regression Model
    regression_model = Sequential([
        Dense(32, activation='relu', input_shape=(X_regression_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    regression_model.compile(optimizer='adam', loss='mse')

    # Train the regression model
    regression_model.fit(X_regression_train, y_regression_train, epochs=500)

    # Evaluate the regression model
    regression_mse = regression_model.evaluate(X_regression_test, y_regression_test)
    regression_model.save('regression.keras')
    print("Mean Squared Error (Regression):", regression_mse)

    loaded_model = load_model('regression.keras')
    predictions = loaded_model.predict(X_prediction)

    [print("Prediction:", predictions[i], "\nActual Data:", y_prediction.iloc[i], '\n') for i in
     range(len(predictions))]
