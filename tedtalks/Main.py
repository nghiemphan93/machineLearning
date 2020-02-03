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

df = pd.read_csv('ted_main.csv')
print(df.head())

import ast
ast.literal_eval()