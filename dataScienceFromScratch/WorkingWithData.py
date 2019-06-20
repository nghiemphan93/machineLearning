from typing import NamedTuple
import datetime


class StockPrice(NamedTuple):
   symbol: str
   date: datetime.date
   closing_price: float

   def is_high_tech(self) -> bool:
      """It's a class, so we can add methods too"""
      return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']


price = StockPrice(symbol='MSFT', date=datetime.date(2018, 12, 14), closing_price=106.03)
print(price.closing_price)
print(type(price.closing_price))
