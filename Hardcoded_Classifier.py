from sklearn import datasets
from numpy.random import permutation

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

print("Your estimated accurate prediction is " + str(num) + "%.")