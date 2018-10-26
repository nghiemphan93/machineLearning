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



def loadData():
   trainPath = "C:/Users/phan/OneDrive - adesso Group/DataSet/electricity/LD2011_2014.txt"
   file = open(trainPath, "rb")
   data = []
   cid = 250
   counter = 0

   array = []

   for line in file:
      if counter == 1000:
         toSave = open("electric.txt", "wb")
         toSave.write(array)
         break
      else:
         array.append(line)
         counter += 1

loadData()

