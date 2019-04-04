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

df = sns.load_dataset("iris")


'''
#sns.pairplot(data=df, hue="species")
g = sns.PairGrid(data=df, hue="species")
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.show()
'''
# sns.pairplot(data=df, hue="species")
# newPoint = pd.DataFrame({"sepal_length": 0,
#                          "sepal_width": 0,
#                          "petal_length": 1.8,
#                          "petal_width": 0.8,
#                          "species": "new point"},
#                         columns=["sepal_length",  "sepal_width",  "petal_length",  "petal_width", "species"])
newPoint = pd.DataFrame(np.array([[0, 0, 2.1, 0.8,"new point"]]),
                        columns=["sepal_length",  "sepal_width",  "petal_length",  "petal_width", "species"])

# df = pd.concat([df, newPoint], axis="rows")
print(df.head(3))

sns.scatterplot(data=df, x="petal_length", y="petal_width")
plt.show()