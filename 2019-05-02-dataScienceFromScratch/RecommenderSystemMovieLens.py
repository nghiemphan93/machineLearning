import csv
import re
import random

from typing import NamedTuple, Tuple, List, Dict

MOVIES = './ml-100k/u.item'
RATINGS = './ml-100k/u.data'


class Rating(NamedTuple):
   userID: str
   movieID: str
   rating: float


with open(MOVIES, encoding='iso-8859-1') as f:
   reader = csv.reader(f, delimiter="|")
   movies = {movieID: title for movieID, title, *_ in reader}
with open(RATINGS, encoding='iso-8859-1') as f:
   reader = csv.reader(f, delimiter='\t')
   ratings = [Rating(userID, movieID, rating) for userID, movieID, rating, _ in reader]

starWarRatings: Dict[str, List[float]] = {movieID: [] for movieID, title in movies.items()
                                          if re.search("Star Wars|Empire Strikes|Jedi", title)}
for rating in ratings:
   if rating.movieID in starWarRatings.keys():
      starWarRatings[rating.movieID].append(rating.rating)

avgRating = []
for movieID, rating in starWarRatings.items():
   tempSum = 0.0
   for i in range(len(rating)):
      tempSum += int(rating[i])
   avgRating.append((tempSum / len(rating), movies[movieID]))

random.seed(0)
random.shuffle(ratings)

split1 = int(len(ratings) * 0.7)
split2 = int(len(ratings) * 0.85)

train = ratings[:split1]
validation = ratings[split1: split2]
test = ratings[split2:]

print(ratings)