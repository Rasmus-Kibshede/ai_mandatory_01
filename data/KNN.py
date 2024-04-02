from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data import data_cleaner, data_handler, data_visualizer, regression_model
from sklearn import preprocessing

#X_test, X_train, X_validation, y_test, y_train, y_validation = data_handler.split_data(X, y)

data = data_handler.load_data('data/modified_dataset.csv')

X = data.drop(['mpg'], axis=1)
y = data['mpg']
X = data_handler.data_encoder(X, ['make', 'model'])


#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y = lab.fit_transform(y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=4)  # You can specify the number of neighbors (k) here

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predicting on the test set
y_pred = knn.predict(X_test)

# Calculating the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

for i in range(len(y_pred)):
    print("Predicted:", y_pred[i], "\tActual:", y_test[i])