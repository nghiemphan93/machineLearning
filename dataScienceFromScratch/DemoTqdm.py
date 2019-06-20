import random
from typing import List

import tqdm


def primesUpTo(n: int) -> List[int]:
   primes = [2]
   with tqdm.trange(3, n) as t:
      t.set_description(desc='clgt')
      for i in t:
         print(t)


primesUpTo(10000)
print("cai leu gi thon")
