import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM

mnist = tf.keras.datasets.mnist

(trainData, trainLabel), (testData, testLabel) = mnist.load_data()

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(trainData[0].shape),
                    return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])
model.fit(trainData, trainLabel,
          epochs=5,
          validation_split=0.2)

result = model.evaluate(testData, testLabel)

print(result)
