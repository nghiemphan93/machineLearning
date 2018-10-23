from pandas.tseries.offsets import MonthEnd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./data/cansim-0800020-eng-6674700030567901031.csv",
                 skiprows=6,
                 skipfooter=9,
                 engine="python")

df["Adjustments"] = pd.to_datetime(df["Adjustments"]) + MonthEnd(1)
df = df.set_index("Adjustments")

print(df)
print(df.info())

df.plot()
plt.show()

