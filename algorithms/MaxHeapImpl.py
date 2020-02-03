from typing import List
import random


def maxHeapify(A: List[float], index: int) -> List[float]:
   pass


def insert(A: List[float], x: float) -> None:
   pass


def max(A: List[float]) -> float:
   pass


def extractMax(A: List[float]) -> float:
   A[0], A[-1] = A[-1], A[0]
   max = A.pop(-1)
   maxHeapify(A, 0)
   return max


def increaseKey(A: List[float], x: float, k: float) -> None:
   pass


def leftChildIndex(i: int) -> int:
   return 2 * i + 1


def rightChildIndex(i: int) -> int:
   return 2 * i + 2


def parentIndex(i) -> int:
   return i / 2


def buildMaxHeap(A: List[float]) -> List[float]:
   for i in range(int(heapSize(A) / 2), -1, -1):
      maxHeapify(A, i)
   return A


def heapSize(A: List[float]) -> int:
   return len(A)


def maxHeapify(A: List[float], i: int) -> None:
   leftIndex = leftChildIndex(i)
   rightIndex = rightChildIndex(i)
   maxIndex = i

   if leftIndex < heapSize(A) and A[i] < A[leftIndex]:
      maxIndex = leftIndex
   if rightIndex < heapSize(A) and A[maxIndex] < A[rightIndex]:
      maxIndex = rightIndex
   if maxIndex != i:
      A[i], A[maxIndex] = A[maxIndex], A[i]
      maxHeapify(A, maxIndex)


def heapSort(A: List[float]) -> List[float]:
   sortedList: List[float] = []

   A = buildMaxHeap(A)
   for i in range(len(A)):
      max = extractMax(A)
      sortedList.insert(0, max)
   return sortedList


def makeRandomList(size: int = 10) -> List[float]:
   A: List[float] = []
   for i in range(size):
      A.append(random.randint(0, 100000))
   return A


A = [16, 4, 10, 14, 7, 9, 3, 2, 8, 1]
B = [9, 4, 10, 2, 7, 16, 3, 14, 8, 1]

C = makeRandomList(size=100000)
print(C)
print(buildMaxHeap(C))
print(heapSort(C))
