import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# Hyperparameter
LEARNING_RATE   = 0.01
BATCH_SIZE      = 1000
EPOCHS          = 10

# Layers
INPUT_SIZE  = 28*28
HL_1        = 1000
HL_2        = 500
N_CLASS     = 10

# Model
model = Sequential()
model.add(Dense(1000, input_dim=INPUT_SIZE, activation="relu"))
model.add(Dense(500, activation="relu"))
model.add(Dropout(rate=0.9))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=["acc"]
)

# Callback
cb = TensorBoard()

# model training
history = model.fit(
    x=mnist.train.images,
    y=mnist.train.labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[cb]
)

# Model testing
score = model.evaluate(
    x=mnist.test.images,
    y=mnist.test.labels
)

print("Score = ", score)

