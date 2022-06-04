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

    timeRules = {
    "dataHubTime":
        {"universeTime":
            {"start_time":[4,0,0],"end_time":[4,15,0]}},
    "performanceTime":
        {"start_time":[4,20,0],"end_time":[4,35,0]}, # Calc Perf prior to optimizing
    "optimizationTime":
        {"start_time":[4,40,0],"end_time":[4,55,0]}, # Optimize Prior to Rebalance
    "updateWeightsTime":
        {'start_time':[5,0,0],'end_time':[5,15,0]},
    "rebalanceTime":
        {"start_time":[6,30,0],"end_time":[6,50,0]}
    } # Rebalance


    def __init__(self):
        pass

    def updateWeightsTimeRules(self,lastUpdate:str) -> bool:
        start = TimeRules.updateWeightsTime["start_time"]
        end = TimeRules.updateWeightsTime["end_time"]

        currDay = date.today().strftime("%Y-%m-%d")
        currTime = datetime.now().time()
        startTime = dt.time(start[0],start[1],start[2])
        endTime = dt.time(end[0], end[1], end[2])

        if np.is_busday(currDay) and startTime <= currTime <= endTime and lastUpdate != currDay:
            return True
        else:
            return False

    def universeTimeRules(self, lastUpdate:str) -> bool:
        start = TimeRules.universeTime["start_time"]
        end = TimeRules.universeTime["end_time"]

        currDay = date.today().strftime("%Y-%m-%d")
        currTime = datetime.now().time()
        startTime = dt.time(start[0],start[1],start[2])
        endTime = dt.time(end[0], end[1], end[2])

        if np.is_busday(currDay) and startTime <= currTime <= endTime and lastUpdate != currDay:
            return True
        else:
            return False

    def rebalanceTimeRules(self, lastUpdate) -> bool:
        start = TimeRules.rebalanceTime["start_time"]
        end = TimeRules.rebalanceTime["end_time"]

        currDay = date.today().strftime("%Y-%m-%d")
        currTime = datetime.now().time()
        startTime = dt.time(start[0],start[1],start[2])
        endTime = dt.time(end[0], end[1], end[2])

        if startTime <= currTime <= endTime and lastUpdate != currDay:# and np.is_busday(currDay):
            return True
        else:
            return False

    def optimizeTimeRules(self, lastUpdate) -> bool:
        start = TimeRules.optimizationTime["start_time"]
        end = TimeRules.optimizationTime["end_time"]

        currDay = date.today().strftime("%Y-%m-%d")
        currTime = datetime.now().time()
        startTime = dt.time(start[0],start[1],start[2])
        endTime = dt.time(end[0], end[1], end[2])

        if np.is_busday(currDay) and startTime <= currTime <= endTime and lastUpdate != currDay:
            return True
        else:
            return False

    def performanceTimeRules(self,lastUpdate)->bool:
        start = TimeRules.performanceTime["start_time"]
        end = TimeRules.performanceTime["end_time"]

        currDay = date.today().strftime("%Y-%m-%d")
        currTime = datetime.now().time()
        startTime = dt.time(start[0],start[1],start[2])
        endTime = dt.time(end[0], end[1], end[2])

        if np.is_busday(currDay) and startTime <= currTime <= endTime and lastUpdate != currDay:
            return True
        else:
            return False
