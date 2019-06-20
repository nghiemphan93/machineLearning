import enum, random


class Kid(enum.Enum):
   BOY = 0
   GIRL = 1


def randomKid() -> Kid:
   return random.choice([Kid.BOY, Kid.GIRL])


bothGirls = 0
olderGirl = 0
eitherGirl = 0
for _ in range(100000):
   older = randomKid()
   younger = randomKid()

   if older == Kid.GIRL:
      olderGirl += 1
   if older == Kid.GIRL and younger == Kid.GIRL:
      bothGirls += 1
   if older == Kid.GIRL or younger == Kid.GIRL:
      eitherGirl += 1
print(f'older girls: {bothGirls/olderGirl}')
print(f'least one girl: {bothGirls/eitherGirl}')
