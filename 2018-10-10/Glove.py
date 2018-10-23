import numpy as np
import os

gloveFolder = "C:/Users/phan/OneDrive - adesso Group/DataSet/glove6b"
gloveFile = "glove.6B.100d.txt"


embeddingIndex = {}
f = open(os.path.join(gloveFolder, gloveFile), encoding="utf8")
for line in f:
   values = line.split()
   word = values[0]
   coeffs = np.asarray(values[1:], dtype="float32")
   embeddingIndex[word] = coeffs
f.close()

embeddingDim = 100
embeddingMatrix = np.zeros((10000, embeddingDim))
