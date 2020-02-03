# region Import
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import datetime as dt

sns.set()
pd.set_option('display.expand_frame_repr', False)
# endregion

filter = 1 / 16 * np.array([
   [1, 2, 1],
   [2, 4, 2],
   [1, 2, 1]
])

matrix = np.array([
   [130, 130, 0],
   [0, 0, 52],
   [28, 52, 0]
])

print(filter)
print(filter.dot(matrix))
