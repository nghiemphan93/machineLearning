import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame
import datetime as dt

matrix = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print(matrix)