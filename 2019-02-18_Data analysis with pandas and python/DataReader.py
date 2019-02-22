# region Import
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
from pandas_datareader import data
# endregion

company="MSFT"
start = "2016-01-01"
end = "2019-12-31"

df = data.DataReader(name=company, data_source="iex",start=start, end=end)
print(df.iloc[200:300])
