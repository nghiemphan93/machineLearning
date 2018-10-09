from keras.models import load_model, Sequential
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models




model_path = "C:/Users/Nghiem Phan/OneDrive - adesso Group/model/dogcat.h5"
img_path = "./cat.1847.jpg"


img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)
plt.imshow(img_tensor[0])
plt.show()

model: Sequential = load_model(model_path)
model.summary()

# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:8]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]

for i in range(16):
   plt.matshow(first_layer_activation[0, :, :, i], cmap='viridis')
   plt.show()