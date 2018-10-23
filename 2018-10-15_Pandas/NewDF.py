from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame


df = pd.DataFrame({"id": [100, 101, 102], "color": ["red", "blue", "red"]},
                  columns=["id", "color"])

array = np.random.rand(4, 2)
df2 = pd.DataFrame(array, columns=["one", "two"])

student = pd.DataFrame({"student": np.arange(100, 110, 1), "test": np.random.randint(60, 101, 10)})

roundSquare = pd.Series(["round", "square"])
print(len(roundSquare))

