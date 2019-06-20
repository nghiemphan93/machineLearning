import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
pd.set_option('display.expand_frame_repr', False)


columns = ['userId', 'itemId', 'rating', 'timestamp']

df = pd.read_csv('u.data', sep='\t', names=columns)
movieTitles = pd.read_csv('Movie_Id_Titles', names=['itemId', 'title'])

df: pd.DataFrame = pd.merge(left=df, right=movieTitles, on='itemId')

# print(df.head())

# print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
# print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['numbRatings'] = df.groupby('title')['rating'].count()


# ratings['numbRatings'].hist(bins=70)
# ratings['rating'].hist(bins=70)
# sns.jointplot(x='rating', y='numbRatings', data=ratings, alpha=0.5)
# plt.show()

print(ratings.head())
moveMat = df.pivot_table(index='userId', columns='title', values='rating')
print(moveMat.head())