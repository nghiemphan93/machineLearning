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

def classify(inX, dataSet, labels, k):
    size = dataSet.shape[0]

    diffMatrix = np.tile(inX, (size,1)) - dataSet      # [point - point, point - point...]
    squareDiffMatrix = diffMatrix**2

    squareDistances = squareDiffMatrix.sum(axis=1)      # [x^2 + y^2, x^2 + y^2...]
    distances = squareDistances**0.5


    sortedDistIndicies = distances.argsort()            # sort distances, save just index []
    labelCount={}                                       #dictionary (label: counter)

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]      # get label given index
        labelCount[voteIlabel] = labelCount.get(voteIlabel,0) + 1   # increase counter according to label

    # sort the dictionary (label: counter) by counter reverse
    # save to array
    sortedClassCount = sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True)

    # return most voted label
    return sortedClassCount
# endregion


# First example
# Label schould be B with 2 votes
data, label = createDataSet()           # dummy data
print(classify([0,0], data, label, 3))



#======================================#
# Second example
# Get data set from iris library and prepare for later processing
iris = datasets.load_iris()
data = iris.data
label = iris.target
labelIndex = iris.target_names


# Take 80 data set for  training
# 20 data set for testing
dataTrain, dataTest, dataTrainLabel, dataTestLabel = train_test_split(data, label, test_size=20)


# Print data
print("data train", dataTrain)
print("train label", dataTrainLabel)
print("data test", dataTest)
print("test label", dataTestLabel)
print()



# Run the nKK algorithm
result = array([])
for i in range(len(dataTest)):
    temp = classify(dataTest[i], dataTrain, dataTrainLabel, 6)
    result = np.append(result, temp[0][0])



# Calculate probability of success
counter = 0
for i in range (len(result)):
    if(dataTestLabel[i] == result[i]):
        counter = counter + 1
success = counter / len(result) * 100.0

# Print results
print("Calculated ", result)
print("Truth      ", dataTestLabel)
print("Succesful: ", success, " %")



