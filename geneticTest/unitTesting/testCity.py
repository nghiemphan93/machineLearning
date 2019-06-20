import unittest
from geneticTSP import City


class TestCity(unittest.TestCase):
   def setUp(self):
      self.cityA = City(15, 20)
      self.cityB = City(45, 100)

   def tearDown(self):
      pass

   def test_calcDistance(self):
      self.assertEqual(self.cityA.distanceTo(nextCity=self.cityB), 85.44003745317531)

   def test__repr__(self):
      self.assertEqual(self.cityA.__repr__(), "(15,20)")


if __name__ == '__main__':
   unittest.main()
