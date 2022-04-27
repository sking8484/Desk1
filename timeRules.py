import os 
import sys 
fpath = os.path.join(os.path.dirname(__file__),'DataHub')
sys.path.append(fpath)

import numpy as np
import datetime as dt
from datetime import datetime, date
from datetime import timedelta
import time

class TimeRules:

    def __init__(self):
        pass

    def universeTimeRules(self, start:list, end:list, lastUpdate:str) -> bool:
        currDay = date.today().strftime("%Y-%m-%d")
        currTime = datetime.now().time()
        startTime = dt.time(start[0],start[1],start[2])
        endTime = dt.time(end[0], end[1], end[2])

        if np.is_busday(currDay) and startTime <= currTime <= endTime and lastUpdate != currDay:
            return True
        else:
            return False

    def rebalanceTimeRules(self, start:list, end:list, lastUpdate) -> bool:
        currDay = date.today().strftime("%Y-%m-%d")
        currTime = datetime.now().time()
        startTime = dt.time(start[0],start[1],start[2])
        endTime = dt.time(end[0], end[1], end[2])

        if np.is_busday(currDay) and startTime <= currTime <= endTime and lastUpdate != currDay:
            return True 
        else:
            return False
