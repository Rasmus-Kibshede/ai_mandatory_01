from data import data_cleaner, data_handler, data_visualizer, regression_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.models import load_model
import numpy as np

def train_model(data):
    X_linear_regression = data.drop(['make', 'model'], axis=1)
    y_linear_regression = data.loc[:, ['make', 'model']]

    y_linear_regression = data_handler.data_encoder(y_linear_regression, ['make', 'model'])

# Split the regression data into training and testing sets
    X_linear_regression_train, X_linear_regression_test, y_linear_regression_train, y_linear_regression_test = \
        train_test_split(X_linear_regression, y_linear_regression, test_size=0.2, random_state=42)
    X_linear_regression_train, X_linear_regression_prediction, y_linear_regression_train, y_linear_regression_prediction\
        = train_test_split(X_linear_regression_train, y_linear_regression_train, test_size=0.5, random_state=42)

    X_linear_regression_train = X_linear_regression_train.astype(np.float32)
    y_linear_regression_train = y_linear_regression_train.astype(np.float32)
    X_linear_regression_test = X_linear_regression_test.astype(np.float32)
    y_linear_regression_test = y_linear_regression_test.astype(np.float32)
    X_linear_regression_prediction = X_linear_regression_prediction.astype(np.float32)
    y_linear_regression_prediction = y_linear_regression_prediction.astype(np.float32)


# Linear Regression Model
    linear_regression_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_linear_regression_train.shape[1],)),  # Specify input shape here
        Dense(32, activation='relu'),
        Dense(len(set(y_linear_regression_train)), activation='softmax')  # Output layer with softmax for classification
    ])

# Compile the linear regression model
    linear_regression_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the linear regression model
    linear_regression_model.fit(X_linear_regression_train, y_linear_regression_train, epochs=50, batch_size=32,
                                validation_split=0.2)

# Evaluate the linear regression model
    linear_regression_loss, linear_regression_accuracy = linear_regression_model.evaluate(X_linear_regression_test,
                                                                                          y_linear_regression_test)
    print("Loss (Linear Regression):", linear_regression_loss)
    print("Accuracy (Linear Regression):", linear_regression_accuracy)

    loaded_model = linear_regression_model.save('linear regression.keras')
    linear_predictions = loaded_model.predict(X_linear_regression_prediction)

    [print("Prediction:", linear_predictions[i], "\nActual Data:", y_linear_regression_prediction.iloc[i], '\n') for i
     in range(len(linear_predictions))]