import operator
from numpy import tile, array, append
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split



# region nKK Algorithm
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']

    return group, labels

def classify(testData, dataSet, labels, k):
    size = dataSet.shape[0]
    distance = calcDistance(dataSet, size, testData)
    sortedDistanceIndex = distance.argsort()
    sortedLabelCount = voteLabel(k, labels, sortedDistanceIndex)

    return sortedLabelCount

def calcDistance(dataSet, size, testData):
    diffMatrix = np.tile(testData, (size, 1)) - dataSet
    squareDiffMatrix = diffMatrix ** 2
    squareDistances = squareDiffMatrix.sum(axis=1)
    distances = np.sqrt(squareDistances)

    return distances

def voteLabel(k, labels, sortedDistIndicies):
    labelCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        labelCount[voteIlabel] = labelCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount

def calcSuccess(result, dataTestLabel):
    counter = 0
    for i in range(len(result)):
        if (dataTestLabel[i] == result[i]):
            counter = counter + 1
            
    return counter / len(result) * 100.0
# endregion


# First example
data, label = createDataSet()           # dummy data
# Test Label should be B with 2 votes
print(classify([0,0], data, label, 3))



#======================================#
# Second example
# Get data set from iris library and prepare for later processing
iris = datasets.load_iris()
data = iris.data
label = iris.target
labelIndex = iris.target_names


# 80 data set for  training
# 20 data set for testing
dataTrain, dataTest, dataTrainLabel, dataTestLabel = train_test_split(data, label, test_size=20)


# Print data
print("data train", dataTrain)
print("train label", dataTrainLabel)
print("data test", dataTest)
print("test label", dataTestLabel)
print()

# Run the nKK algorithm
result = np.array([])
for i in range(len(dataTest)):
    temp = classify(dataTest[i], dataTrain, dataTrainLabel, 6)
    result = np.append(result, temp[0][0])

# Calculate Success
success = calcSuccess(result, dataTestLabel)

# Print results
print("Calculated ", result)
print("Truth      ", dataTestLabel)
print("Successful: ", success, " %")



