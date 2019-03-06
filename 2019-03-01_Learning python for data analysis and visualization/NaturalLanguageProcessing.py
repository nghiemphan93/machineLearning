# region Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
import statsmodels.api as sm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import sklearn.metrics as metrics
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
# endregion

#messages = [line.rstrip() for line in open("./data/SMSSpamCollection")]
df = pd.read_csv("./data/SMSSpamCollection", sep="\t", names=["label", "message"])
df["length"] = df["message"].apply(len)

#sns.countplot(x="length", data=df, hue="label")
#sns.distplot(a=df["length"], kde=True)
#sns.catplot(x="length", data=df, kind="count", hue="label")
#df.hist(column="length", by="label")
#plt.show()

messTemp = "Sample message! notice: it has puncuation"

def textPreprocess(text):
   nopunc = [char for char in text if char not in string.punctuation]
   nopunc = "".join(nopunc)
   noStopWords = [word for word in nopunc.split() if word not in stopwords.words("english")]
   return noStopWords

bowTransformer = CountVectorizer(analyzer=textPreprocess)
bowTransformer.fit(df["message"].head())
print(bowTransformer.transform(df["message"].head(1)))

print(df["message"].head(1))
print(bowTransformer.get_feature_names()[5])
print(bowTransformer.get_feature_names()[6])
print(bowTransformer.get_feature_names()[26])