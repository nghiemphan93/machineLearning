import operator


from numpy import tile, array, append
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split



def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify(inX, dataSet, labels, k):
    size = dataSet.shape[0]

    diffMatrix = np.tile(inX, (size,1)) - dataSet      # [point - point, point - point...]
    squareDiffMatrix = diffMatrix**2

    squareDistances = squareDiffMatrix.sum(axis=1)  # [x^2 + y^2, x^2 + y^2...]
    distances = squareDistances**0.5


    sortedDistIndicies = distances.argsort()        # sort distances, save just index []
    labelCount={}                                   #dictionary (label: counter)

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # get label given index
        labelCount[voteIlabel] = labelCount.get(voteIlabel,0) + 1   # increase counter according to label

    # sort the dictionary (label: counter) by counter reverse
    # save to array
    sortedClassCount = sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True)

    # return most voted label
    return sortedClassCount


data, label = createDataSet()
print(classify([0,0], data, label, 3))



iris = datasets.load_iris()
data = iris.data
label = iris.target
labelIndex = iris.target_names


dataTrain, dataTest, dataTrainLabel, dataTestLabel = train_test_split(data, label, test_size=20)



print("data train", dataTrain)
print("train label", dataTrainLabel)
print("data test", dataTest)
print("data test label", dataTestLabel)




result = array([])
for i in range(len(dataTest)):
    temp = classification(dataTest[i], dataTrain, dataTrainLabel, 7)
    result = np.append(result, temp[0][0])



counter = 0
for i in range (len(result)):
    if(dataTestLabel[i] == result[i]):
        counter = counter + 1
success = counter / len(result) * 100.0


print("Calculated ", result)
print("Truth      ", dataTestLabel)
print("Succesful: ", success, " %")



