from keras import preprocessing
import numpy as np
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.callbacks import TensorBoard

def preprocessData():
   # Load data
   df = np.loadtxt("./audiobook.csv", delimiter=",")
   rawInput  = df[:, 1:-1]
   targets        = df[:, -1:]

   # Balance data
   oneTargetCounter  = np.sum(targets)
   zeroTargetCounter = 0
   indicesToRemove = []

   for i in range(targets.shape[0]):
      if targets[i] == 0:
         zeroTargetCounter += 1
         if zeroTargetCounter > oneTargetCounter:
            indicesToRemove.append(i)

   balancedUnscaledInputs  = np.delete(rawInput, indicesToRemove, axis=0)
   balancedTarget          = np.delete(targets, indicesToRemove, axis=0)


   # Scale inputs
   scaledInput    = preprocessing.normalize(balancedUnscaledInputs)
   scaledTarget   = balancedTarget

   # Shuffle data
   shuffledIndices   = np.arange(scaledInput.shape[0])
   np.random.shuffle(shuffledIndices)

   shuffledInputs    = scaledInput[shuffledIndices]
   shuffledTargets   = scaledTarget[shuffledIndices]

   # Split train and test data
   sampleCount    = shuffledInputs.shape[0]
   trainCount     = int(0.9 * sampleCount)
   testCount      = int(0.1 * sampleCount)

   trainData   = shuffledInputs[:trainCount]
   trainTarget = shuffledTargets[:trainCount]

   testData = shuffledInputs[trainCount:]
   testTarget = shuffledTargets[trainCount:]

   np.savez("audiobookData.npz",
            trainData=trainData,
            trainTarget=trainTarget,
            testData=testData,
            testTarget=testTarget)

def loadData():
   data = np.load("audiobookData.npz")
   trainData   = data["trainData"]
   trainTarget = data["trainTarget"]
   testData    = data["testData"]
   testTarget  = data["testTarget"]
   return trainData, trainTarget, testData, testTarget


# Load data
trainData, trainTarget, testData, testTarget = loadData()
trainTarget = to_categorical(trainTarget)
testTarget = to_categorical(testTarget)


model = Sequential()
model.add(Dense(200, activation="relu", input_shape=(10,)))


model.add(Dense(200, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(200, activation="relu"))

model.add(Dense(2, activation="softmax"))
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(trainData,
          trainTarget,
          epochs=8,
          batch_size=100,
          validation_split=0.2)

test_loss, test_acc = model.evaluate(testData, testTarget)
print('test_acc:', test_acc)








