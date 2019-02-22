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

day = dt.date(2016, 4, 12)
timestamp = pd.Timestamp(day)

dateRange = pd.date_range(start="2016-01-01", end="2016-01-15", freq="15MIN")
print(dateRange)
