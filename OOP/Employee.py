class Employee:
   def __init__(self, first: str, last: str, pay: float):
      self.first = first
      self.last = last
      self.pay = pay

   def fullname(self):
      return f'${self.first} ${self.last}'


