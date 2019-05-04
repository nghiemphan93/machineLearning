from collections import defaultdict, Counter
from typing import List, Dict, Any

users = [
   {"id": 0, "name": "Hero"},
   {"id": 1, "name": "Dunn"},
   {"id": 2, "name": "Sue"},
   {"id": 3, "name": "Chi"},
   {"id": 4, "name": "Thor"},
   {"id": 5, "name": "Clive"},
   {"id": 6, "name": "Hicks"},
   {"id": 7, "name": "Devin"},
   {"id": 8, "name": "Kate"},
   {"id": 9, "name": "Klein"}
]
friendshipPairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                   (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
friendShipDict: Dict[int, List[int]] = {user['id']: [] for user in users}

# loop over populate friendship dict
for i, j in friendshipPairs:
   friendShipDict[i].append(j)
   friendShipDict[j].append(i)


def numberOfFriends(user):
   return len(friendShipDict[user['id']])


totalConnections = sum(numberOfFriends(user) for user in users)

numbFriendsById = [(user['id'], numberOfFriends(user)) for user in users]
numbFriendsById.sort(key=lambda pair: pair[1], reverse=True)


def calcFriendOfAFriend(user):
   return {friendOfFriendId for friendId in friendShipDict[user['id']]
           for friendOfFriendId in friendShipDict[friendId]
           if friendOfFriendId != user['id'] and friendOfFriendId not in friendShipDict[user['id']]}


interests = [
   (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
   (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
   (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
   (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
   (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
   (3, "statistics"), (3, "regression"), (3, "probability"),
   (4, "machine learning"), (4, "regression"), (4, "decision trees"),
   (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
   (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
   (6, "probability"), (6, "mathematics"), (6, "theory"),
   (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
   (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
   (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
   (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]


def dataScientistWhoLike(targetInterest):
   return [userId for userId, interest in interests
           if interest == targetInterest]


interest2users = defaultdict(list)
for userId, interest in interests:
   interest2users[interest].append(userId)

user2interests = defaultdict(list)
for userId, interest in interests:
   user2interests[userId].append(interest)


def most_common_interests_with(user):
   return Counter(
      interested_user_id
      for interest in user2interests[user["id"]]
      for interested_user_id in interest2users[interest]
      if interested_user_id != user["id"])


salariesAndTenures = [(83000, 8.7), (88000, 8.1),
                      (48000, 0.7), (76000, 6),
                      (69000, 6.5), (76000, 7.5),
                      (60000, 2.5), (83000, 10),
                      (48000, 1.9), (63000, 4.2)]


def tenureBucket(tenure: float) -> str:
   if tenure < 2:
      return "less than two"
   elif tenure < 5:
      return "between two and five"
   else:
      return "more than five"


bucket2salaries = defaultdict(list)
for salary, tenure in salariesAndTenures:
   bucket2salaries[tenureBucket(tenure)].append(salary)

bucket2avgSalary = {bucket: sum(salaries) / len(salaries) for bucket, salaries in bucket2salaries.items()}
print(bucket2avgSalary)
