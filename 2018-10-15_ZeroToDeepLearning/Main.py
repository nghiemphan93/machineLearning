from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)


img = Image.open("./data/iss.jpg")
img = np.asarray(img)

print(img.shape)
img = img / 255

plt.imshow(img)
plt.show()