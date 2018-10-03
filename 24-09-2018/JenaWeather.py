import os
import numpy as np
import matplotlib.pyplot as plt


def loadData(fileName):
    file = open(fileName, encoding="utf-8")
    data = file.read()
    file.close()

    # filter header and lines
    lines = data.split("\n")
    header = lines[0].split(",")
    lines = lines[1:]

    # convert lines to numpy array
    tensor = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        lineAsArray = [temp for temp in line.split(",")[1:]]
        tensor[i] = lineAsArray

    '''
    temp = tensor[:, 1]
    plt.plot(range(17280), temp[:17280])
    plt.show()
    '''
    return tensor


fileName = "C:/Users/phan/Downloads/DataSet/weather/jena_climate_2009_2016.csv"
data = loadData(fileName)


mean    = data[:200000].mean(axis=0)
data    -= mean
std     = data[:200000].std(axis=0)
data    /= std

print(std)
