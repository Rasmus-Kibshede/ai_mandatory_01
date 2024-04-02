from keras import Sequential
from keras.models import load_model
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


def train_model(data):
    X = data.drop(columns=['make', 'model'])  # Features
    y = data['make']  # Target

    # Convert categorical 'make' labels into numerical values
    label_encoder, y = encode_data(y)

    # Normalize the features
    X = scale_data(X)

    # Split the data into train and test sets
    X_test, X_train, X_validation, y_test, y_train, y_validation = split_data(X, y)

    # Model Building
    model = build_model(X_train, label_encoder, 32, 2)

    # Model Training
    fit_model(X_train, model, y_train, 2)

    # Model Evaluation
    evaluate_model(X_test, model, y_test)

    result(X_validation, label_encoder, y_validation)


def result(X_validation, label_encoder, y_validation):
    loaded_model = load_model('linear regression.keras')
    linear_predictions = loaded_model.predict(X_validation)
    prediction = np.array(linear_predictions)
    decoded_y_validation = label_encoder.inverse_transform(y_validation)

    [print("Predicted Class Label:", label_encoder.classes_[np.argmax(prediction[i])],
           "\nActual Class Label (Decoded):", decoded_y_validation[i], '\n')
     for i in range(len(prediction))]


def evaluate_model(X_test, model, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test Accuracy:', test_acc)
    model.save('linear regression.keras')


def fit_model(X_train, model, y_train, epochs):
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)


def build_model(X_train, label_encoder, nodes, layers):
    model = Sequential()
    model.add(Dense(nodes, activation='relu', input_shape=(X_train.shape[1],)))
    [model.add(Dense(nodes, activation='relu')) for _ in range(layers)]
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    return X_test, X_train, X_validation, y_test, y_train, y_validation


def encode_data(y):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return label_encoder, y


def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
