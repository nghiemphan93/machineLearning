from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import numpy as np
import math

usersInterests = [
   ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
   ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
   ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
   ["R", "Python", "statistics", "regression", "probability"],
   ["machine learning", "regression", "decision trees", "libsvm"],
   ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
   ["statistics", "probability", "mathematics", "theory"],
   ["machine learning", "scikit-learn", "Mahout", "neural networks"],
   ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
   ["Hadoop", "Java", "MapReduce", "Big Data"],
   ["statistics", "R", "statsmodels"],
   ["C++", "deep learning", "artificial intelligence", "probability"],
   ["pandas", "R", "Python"],
   ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
   ["libsvm", "regression", "support vector machines"]
]

# print(users_interests)
# popular_interests = Counter(interest for user_interests in users_interests
#                             for interest in user_interests)
users = [i for i in range(len(usersInterests))]
interests = [interest for userInterest in usersInterests
             for interest in userInterest]
# for userInterest in users_interests:
#    for interest in userInterest:
#       interests.append(interest)
popularInterest = Counter(interests)


# print(popularInterest.most_common(5))


def mostPopularNewInterest(popularInterest: Counter,
                           oneUserInterests: List[str],
                           maxResults: int = 5) -> List[Tuple[str, int]]:
   recommendations: List[Tuple[str, int]] = []
   for interest, frequency in popularInterest.most_common():
      if interest not in oneUserInterests:
         print(interest)
         recommendations.append((interest, frequency))
   return recommendations[:maxResults]


uniqueInterests = sorted(set([interest for eachUserInterests in usersInterests
                              for interest in eachUserInterests]))


def makeUserInterestVector(oneUsersInterests: List[str]) -> List[int]:
   return [1 if interest in oneUsersInterests
           else 0
           for interest in uniqueInterests]


def calcCosineSimilar(interestVectorA: List[int], interestVectorB: List[int]) -> float:
   dot: float = np.dot(interestVectorA, interestVectorB)
   norm: float = np.linalg.norm(interestVectorA) * np.linalg.norm(interestVectorB)
   return dot / norm


def norm(vector: List[int]):
   sum = 0
   for i in range(len(vector)):
      sum += vector[i] ** 2
   return math.sqrt(sum)


usersInterestVectors = [makeUserInterestVector(eachUserInterests) for eachUserInterests in usersInterests]
userSimilarities: List[List[float]] = [
   [calcCosineSimilar(userVectorA, userVectorB) for userVectorA in usersInterestVectors]
   for userVectorB in usersInterestVectors]


def mostSimilarUsersTo(userIndex: int) -> List[Tuple[int, float]]:
   pairs: List[Tuple[int, float]] = [(otherUserIndex, cosineValue)
                                     for otherUserIndex, cosineValue in
                                     enumerate(userSimilarities[userIndex])
                                     if (otherUserIndex != userIndex and cosineValue > 0)]
   return sorted(pairs,
                 key=lambda pair: pair[1],
                 reverse=True)


def userBasedSuggestions(userIndex: int, includeCurrentInterest: bool = False) -> List[Tuple[str, float]]:
   # Sum up simi
   suggestions: Dict[str, float] = defaultdict(float)
   for otherUserIndex, similarity in mostSimilarUsersTo(userIndex):
      for interest in usersInterests[otherUserIndex]:
         suggestions[interest] += similarity

   # sort by similar
   suggestions = sorted(suggestions.items(), key=lambda pair: pair[1], reverse=True)
   # print(suggestions)

   # exclude maybe
   if includeCurrentInterest:
      return suggestions
   else:
      return [(interest, similarity) for interest, similarity in suggestions
              if interest not in usersInterests[userIndex]]


# print(userBasedSuggestions(0))

def makeInterestUserVector(interest: str) -> List[int]:
   return [1 if interest in oneUserInterests
           else 0
           for userIndex, oneUserInterests in enumerate(usersInterests)]


# interestUserMatrix = [makeInterestUserVector(interest)for interest in uniqueInterests]
interestUserMatrix = [[1 if interest in oneUserInterests
                       else 0
                       for userIndex, oneUserInterests in enumerate(usersInterests)]
                      for interest in uniqueInterests]

interestSimilarities = [[calcCosineSimilar(interestVectorA, interestVectorB)
                         for interestVectorA in interestUserMatrix]
                        for interestVectorB in interestUserMatrix]


def mostSimilarInterestsTo(interestIndex: int) -> List[Tuple[str, float]]:
   mostSimilarInterests = [(uniqueInterests[otherInterestIndex], similarity)
                           for otherInterestIndex, similarity in
                           enumerate(interestSimilarities[interestIndex])
                           if otherInterestIndex != interestIndex and similarity > 0]
   mostSimilarInterests = sorted(mostSimilarInterests, key=lambda pair: pair[-1], reverse=True)
   return mostSimilarInterests


def itemBasedSuggestions(userIndex: int, inclueCurrentInterest: bool = False) -> List[Tuple[str, float]]:
   # add up similar interest
   suggestions: Dict[str, float] = defaultdict(float)
   for interest in usersInterests[userIndex]:
      interestIndex = uniqueInterests.index(interest)
      for similarInterest, similarity in mostSimilarInterestsTo(interestIndex):
         suggestions[similarInterest] += similarity

   # sort by weight
   suggestions = sorted(suggestions.items(), key=lambda pair: pair[-1], reverse=True)

   # include or exclude already interests
   if inclueCurrentInterest:
      return suggestions
   else:
      return [(interest, similarity) for interest, similarity in suggestions
              if interest not in usersInterests[userIndex]]


print(itemBasedSuggestions(0))
