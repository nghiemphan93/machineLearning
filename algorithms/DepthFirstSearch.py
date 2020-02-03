from collections import defaultdict
from typing import Dict, List, DefaultDict

from algorithms.BreadthFirstSearchAdj import createAdj, Vertices

adj = createAdj()
V = Vertices

# visitedVertices = []
parents: DefaultDict[str, str] = defaultdict()


def dfsVisit(adj: Dict[str, List[str]], currentVertex: str):
   for neighbor in adj[currentVertex]:
      if neighbor not in parents:
         parents[neighbor] = currentVertex
         dfsVisit(adj, neighbor)


def dfs(V: List[str], adj: Dict[str, List[str]]):
   for vertexToVisit in V:
      if vertexToVisit not in parents:
         parents[vertexToVisit] = None
         dfsVisit(adj, vertexToVisit)


dfs(Vertices, adj)
print(parents)
