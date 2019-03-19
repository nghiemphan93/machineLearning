import unittest
from random import Random
from typing import List
from genetic.TSP.City import City
from genetic.TSP.DNA import DNA
from genetic.TSP.Helper import Helper


class TestDNA(unittest.TestCase):
   def setUp(self):
      self.route = [City(432, 197), City(388, 455), City(215, 20), City(132, 494), City(261, 248)]
      self.random = Random(0)
      self.dna = DNA(self.route)
      self.dna.random = self.random

   def tearDown(self):
      pass

   def test_shuffleRoute(self):
      self.assertEqual(self.dna.shuffleRoute(), 0)



if __name__ == '__main__':
   unittest.main()
