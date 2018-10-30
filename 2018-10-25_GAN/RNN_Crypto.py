import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Bidirectional, Flatten, Dense, Embedding, LSTM, CuDNNLSTM, Bidirectional, GRU, CuDNNGRU, SpatialDropout1D, Dropout, Conv2D, Conv1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')
pd.set_option('display.expand_frame_repr', False)


trainPath = "C:/Users/phan/OneDrive - adesso Group/DataSet/crypto_data"


mainDf = pd.DataFrame()

ratios = ["BTC-USD.csv", "LTC-USD.csv", "ETH-USD.csv", "BCH-USD.csv"]
for ratio in ratios:
   file = os.path.join(trainPath, ratio)
   df = pd.read_csv(file, names=["time", "low", "high", "open", "close", "volume"])
   df.rename(columns={"close": f"{ratio[:3]}_close",
                      "volume": f"{ratio[:3]}_volume"},
             inplace=True)
   df.set_index("time", inplace=True)
   df = df[[f"{ratio[:3]}_close", f"{ratio[:3]}_volume"]]

   if len(mainDf) == 0:
      mainDf = df
   else:
      mainDf = mainDf.join(df)


mainDf.dropna(inplace=True)
print(mainDf)
print(mainDf.info())

seqLen = 60
futurePeriodPredict = 3
ratioToPredict = "LTC-USD"

