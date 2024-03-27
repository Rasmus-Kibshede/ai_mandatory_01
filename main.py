from data import data_cleaner, data_handler, data_visualizer, regression_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

##TODO MAIN SKAL rydes op og kun kalde på metoder i regression, og linear regression.

# Loads our modified data
data = data_handler.load_data('data/modified_dataset.csv')
# data_visualizer.visualize_data(data)

# Should encoded data be saved as well?
# data = data_handler.data_encoder(data, ['make', 'model'])
##TODO REGRESSION

X_regression = data.drop(['mpg'], axis=1)
y_regression = data['mpg']
X_regression = data_handler.data_encoder(X_regression, ['make', 'model'])

#Splits original dataset into 70/30
X_regression_train, X_temp, y_regression_train, y_temp = train_test_split(X_regression, y_regression,
                                                                          test_size=0.3, random_state=42)
#Splits test set to 50/50
X_regression_test, X_prediction, y_regression_test, y_prediction = train_test_split(X_temp, y_temp,
                                                                                    test_size=0.5, random_state=42)

X_regression_train = X_regression_train.astype(np.float32)
y_regression_train = y_regression_train.astype(np.float32)
X_regression_test = X_regression_test.astype(np.float32)
y_regression_test = y_regression_test.astype(np.float32)
X_prediction = X_prediction.astype(np.float32)

# Regression Model
regression_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_regression_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model
regression_model.compile(optimizer='adam', loss='mse')

# Train the regression model
print(X_regression_train.shape, ' Size ', y_regression_train.shape)
regression_model.fit(X_regression_train, y_regression_train, epochs=1, batch_size=32, validation_split=0.2)

# Evaluate the regression model
regression_mse = regression_model.evaluate(X_regression_test, y_regression_test)
regression_model.save('regression.keras')
print("Mean Squared Error (Regression):", regression_mse)

loaded_model = load_model('regression.keras')
predictions = loaded_model.predict(X_prediction)

[print("Prediction:", predictions[i], "\nActual Data:", y_prediction.iloc[i], '\n') for i in range(len(predictions))]



##TODO linear REGRESSION
# Data preprocessing
X = data.drop(columns=['make', 'model'])  # Features
y = data['make']  # Target

# Convert categorical 'make' labels into numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#X = label_encoder.fit_transform(X)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# Model Building
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Model Compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model Training
history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2)

# Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)
model.save('linear regression.keras')

loaded_model = load_model('linear regression.keras')

#linear_regression_model.save('linear regression.keras')
linear_predictions = loaded_model.predict(X_validation)
prediction = np.array(linear_predictions)

# Find the index of the class with the highest probability
predicted_class_index = np.argmax(prediction)

decoded_y_validation = label_encoder.inverse_transform(y_validation)

for i in range(len(prediction)):
    predicted_class_index = np.argmax(prediction[i])
    print("Predicted Class Label:", label_encoder.classes_[predicted_class_index])
    print("Actual Class Label (Decoded):", decoded_y_validation[i], '\n')


def prepare_data():
    original_data = data_handler.load_data('data/cars.csv')
    data_cleaner.clean(original_data)
