from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame


df: DataFrame = pd.read_csv("http://bit.ly/drinksbycountry")

df = df.groupby("continent").mean()
df.plot(kind="bar")
plt.show()
print(df)
