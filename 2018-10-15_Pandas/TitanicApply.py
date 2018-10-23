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

df["lastName"] = df["Name"].str.split(",").apply(lambda x: x[0])
df["firstName"] = df["Name"].str.split(",").apply(lambda x: x[1])


drink = pd.read_csv("http://bit.ly/drinksbycountry")

print(drink.loc[:, "beer_servings": "wine_servings"].apply(np.argmax, axis=0))
