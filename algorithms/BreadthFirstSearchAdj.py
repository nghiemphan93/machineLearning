from collections import defaultdict
from typing import Dict, List, DefaultDict


def createAdj() -> Dict:
   graph: DefaultDict[str, List[str]] = defaultdict(list)

   graph['a'].append('s')
   graph['a'].append('z')

   graph['z'].append('a')

   graph['s'].append('a')
   graph['s'].append('x')

   graph['x'].append('d')
   graph['x'].append('c')
   graph['x'].append('s')

   graph['d'].append('c')
   graph['d'].append('x')

   graph['c'].append('d')
   graph['c'].append('f')
   graph['c'].append('x')
   graph['c'].append('v')

   graph['f'].append('c')
   graph['f'].append('d')
   graph['f'].append('v')

   graph['v'].append('c')
   graph['v'].append('f')

   return graph


adj = createAdj()

Vertices = ['a', 'z', 's', 'x', 'd', 'c', 'f', 'v']


def breadthFirstSearch(adj: Dict[str, List[str]], startVertex: str):
   visitedVertices = {startVertex: 0}
   parents = {startVertex: None}
   deepLevel = 1
   verticesToVisit = [startVertex]

   while verticesToVisit:
      neighborsOfNeighbors = []

      for vertexToVisit in verticesToVisit:
         for neighbor in adj[vertexToVisit]:
            if neighbor not in visitedVertices.keys():
               visitedVertices[neighbor] = deepLevel
               parents[neighbor] = vertexToVisit
               neighborsOfNeighbors.append(neighbor)
      verticesToVisit = neighborsOfNeighbors
      deepLevel += 1
   print(visitedVertices)


if __name__ == '__main__':
   for key, value in adj.items():
      print(key, value)
   breadthFirstSearch(adj, 's')
