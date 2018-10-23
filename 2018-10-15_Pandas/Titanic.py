from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from sklearn.linear_model import LogisticRegression



df: DataFrame = pd.read_csv("http://bit.ly/kaggletrain")

print(df.info())

featureCols = ["Pclass", "Parch"]
X = df.loc[:, featureCols]
y = df["Survived"]

print(y.shape)
