import matplotlib.pyplot as plt
import keras.applications
from keras import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

vgg16Model: Model = keras.applications.vgg16.VGG16()

print(len(vgg16Model.layers))

model = Sequential()
for layer in vgg16Model.layers[:-1]:
    model.add(layer)

model.summary()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation="softmax"))
model.summary()



# All images will be rescaled by 1./255
#train_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_dir = "C:/Users/phan/Downloads/DataSet/DogCat/cats_and_dogs_small/train"
validation_dir = "C:/Users/phan/Downloads/DataSet/DogCat/cats_and_dogs_small/validation"
test_dir = "C:/Users/phan/Downloads/DataSet/DogCat/cats_and_dogs_small/test"

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(224, 224),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical')

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["acc"])
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              validation_data=validation_generator,
                              validation_steps=50,
                              epochs=1)


model.save("dogCatFineTune.h5")


predictions = model.evaluate_generator(test_generator, steps=34, verbose=1)
print(predictions)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()