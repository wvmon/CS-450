from sklearn import datasets, cross_validation, preprocessing
from numpy.random import permutation
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Show the data (the attributes of each instance)
iris = datasets.load_iris()

# Randomize my data
indices = permutation(len(iris.data))

# Split the data for the training data set and target set
training_data_set = iris.data[indices[0:100:1]]
training_target_set = iris.target[indices[0:100:1]]

# Split the data for the testing data set and target set
testing_data_set = iris.data[indices[100:150:1]]
testing_target_set = iris.target[indices[100:150:1]]


car_data = pd.read_csv('car.data.txt')
print (car_data)

data = car_data[['data', 'data1', 'data2', 'data3', 'data4', 'data5']]
target = car_data['target']

X = np.array(car_data.drop(['target'], 1))
y = np.array(car_data['target'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
accuracy = classifier.score(X_test, y_test)
print(predictions)
print(accuracy)

def standardizing(X_train, X_test):
    std_scale = preprocessing.StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)

# Parent class
class HardCoded(object):

    def train(self, training_dataset, training_target):
        pass

    def predict(self, testing_data):
        predictions = []
        for i in range(len(testing_data)):
            # assign which flower you want to predict
            predictions.append(2)
        return predictions


hardCoder = HardCoded()
count = 0

accuracy = hardCoder.predict(testing_data_set)

# Test the accuracy of your results
for i in range(len(accuracy)):
    if accuracy[i] ==  testing_target_set[i]:
        count += 1

num = (count/len(testing_target_set)) * 100

#print("Your estimated accurate prediction is " + str(num) + "%.")