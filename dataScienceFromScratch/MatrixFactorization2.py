import csv
import random
from typing import NamedTuple

MOVIES = 'ml-100k/u.item'
RATINGS = 'ml-100k/u.data'

class Rating(NamedTuple):
   userId: str
   movieId: str
   rating: float


with open(file=MOVIES, encoding='iso-8859-1') as f:
   reader = csv.reader(f, delimiter='|')
   movies = {movieId: title for movieId, title, *_ in reader}
with open(file=RATINGS, encoding='iso-8859-1') as f:
   reader = csv.reader(f, delimiter='\t')
   ratings = [Rating(userId, movieId, float(rating)) for userId, movieId, rating, _ in reader]

random.seed(0)
random.shuffle(ratings)

split1 = int(len(ratings) * 0.7)
split2 = int(len(ratings) * 0.85)

train = ratings[:split1]
validation = ratings[split1:split2]
test = ratings[split2:]

avgRating = sum([rating.rating for rating in ratings]) / len(ratings)
stdErrorRating = sum([(rating.rating - avgRating)**2 for rating in test]) / len(test)

print(avgRating)
print(stdErrorRating)

print(train)