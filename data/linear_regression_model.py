from keras import Sequential
from keras.models import load_model
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


def train_model(data):
    ##TODO NAVNGIVNING SKAL LAVES OM!
    ##TODO Skal have en save model metode
    ##DATA SPLIT metode
    ##Model load metode osv.
    # Data preprocessing
    X = data.drop(columns=['make', 'model'])  # Features
    y = data['make']  # Target

    # Convert categorical 'make' labels into numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # X = label_encoder.fit_transform(X)

    # Normalize the features
    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Model Building
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
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

    # linear_regression_model.save('linear regression.keras')
    linear_predictions = loaded_model.predict(X_validation)
    prediction = np.array(linear_predictions)

    # Find the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)

    decoded_y_validation = label_encoder.inverse_transform(y_validation)

    for i in range(len(prediction)):
        predicted_class_index = np.argmax(prediction[i])
        print("Predicted Class Label:", label_encoder.classes_[predicted_class_index])
        print("Actual Class Label (Decoded):", decoded_y_validation[i], '\n')
