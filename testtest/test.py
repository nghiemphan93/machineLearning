# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = 8, 4
# sns.set()
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion


df = pd.read_csv("./spielen.csv", delimiter=" ")


def calEntropy(df: pd.DataFrame, column: str, target: str):
   df = df[[column, target]]
   uValues = df[column].unique()
   uTarget = df[target].unique()

   entropyList = []
   for valueName in uValues:
      entropySingle = 0
      neinNumb = df[(df[column] == valueName) & (df[target] == "Nein")][target].count()
      jaNumb = df[(df[column] == valueName) & (df[target] == "Ja")][target].count()
      totalNumb = neinNumb + jaNumb
      entropySingle = -(neinNumb/jaNumb)*(np.log(neinNumb/jaNumb) / np.log(2))
      print(entropySingle)


      print(valueName, " ", neinNumb, " ", jaNumb)




column = "vorhersage"
target = "spielen"
calEntropy(df, column, target)
print(df.sort_values(by=column))
# df = df[[column, target]]
# print(df[df[column] == "Sonnig"])