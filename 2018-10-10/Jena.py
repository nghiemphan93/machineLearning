import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder = "C:/Users/phan/OneDrive - adesso Group/DataSet/weather"
fileName = os.path.join(folder, "jena_climate_2009_2016.csv")

# Preprocess data
df = pd.read_csv(fileName)
df = df.drop(["Date Time"], axis=1)
data = df.values

# Normalize data
mean = data.mean(axis=0)
data -= mean
std = data.std(axis=0)
data /= std

print(data)