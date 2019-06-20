import csv, re, random

from typing import NamedTuple, Dict, List

MOVIES = 'ml-100k/u.item'
RATINGS = 'ml-100k/u.data'


class Rating(NamedTuple):
   userId: str
   movieId: str
   rating: float


with open(MOVIES, encoding='iso-8859-1') as f:
   reader = csv.reader(f, delimiter='|')
   movies = {movieId: title for movieId, title, *_ in reader}

with open(RATINGS, encoding='iso-8859-1') as f:
   reader = csv.reader(f, delimiter='\t')
   ratings = [Rating(userId, movieId, float(rating)) for userId, movieId, rating, _ in reader]

# find star war movieID from movies.title
starWarsRatings: Dict[str, List[float]] = {movieId: [] for movieId, title in movies.items()
                                           if re.search(pattern="Star Wars|Empire Strikes|Jedi", string=title)}

# iterate over ratings and collect ratings for star war
for rating in ratings:
   if rating.movieId in starWarsRatings.keys():
      starWarsRatings[rating.movieId].append(rating.rating)

# sum it up and average
avgRatings = [(sum(ratings) / len(ratings), movieId) for movieId, ratings in starWarsRatings.items()]
avgRatings.sort(reverse=True)
for avgRating, movieId in avgRatings:
   print(f'{avgRating:.2f} {movies[movieId]}')

random.seed(0)
random.shuffle(ratings)

split1 = int(len(ratings) * 0.7)
split2 = int(len(ratings) * 0.85)

train = ratings[:split1]
validation = ratings[split1:split2]
test = ratings[split2:]

avgRating = sum(rating.rating for rating in train) / len(train)
baselineError = sum((rating.rating - avgRating) ** 2 for rating in train) / len(train)

from dataScienceFromScratch.deeplearning import random_tensor

EMBEDDING_DIM = 2
# find unique ids from ratings
userIds = set(rating.userId for rating in ratings)
movieIds = set(rating.movieId for rating in ratings)

# create a random vector per id
userVectors = {userId: random_tensor(EMBEDDING_DIM) for userId in userIds}
movieVectors = {movieId: random_tensor(EMBEDDING_DIM) for movieId in movieIds}

print(userVectors)
