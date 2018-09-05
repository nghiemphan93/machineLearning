import time
from enum import Enum

#=================================#
# Simulate color sequence of a traffic light with Class and Enum
#=================================#


class LightColor(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3

class TrafficLight:
    def __init__(self, color):
        self.color = LightColor.GREEN
        self.colorLoop = [LightColor.RED, LightColor.GREEN, LightColor.YELLOW]

    def changeColor(self):
        currIndex = self.colorLoop.index(self.color)

        if(currIndex == len(self.colorLoop)-  1 ):
            self.color = self.colorLoop[0]
        else:
            self.color = self.colorLoop[currIndex + 1]

ampel = TrafficLight(LightColor.GREEN)

# Sequence
while(True):
    print(ampel.color)
    ampel.changeColor()
    time.sleep(1)
