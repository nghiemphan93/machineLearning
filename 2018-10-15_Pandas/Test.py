from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')


columns = ["userId", "age", "gender", "occupation", "zip"]
df = pd.read_table("http://bit.ly/movieusers",
                   sep="|",
                   header=None,
                   names=columns)
print(df)