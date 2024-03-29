import os
import sys

from pandas import array
fpath = os.path.join(os.path.dirname(__file__),'DataHub')
sys.path.append(fpath)

import numpy as np
import datetime as dt
from datetime import datetime, date
from datetime import timedelta
import time

class TimeRules:
    def __init__(self):
        self.timeRules = {
            "dataHub":
                {"maintainUniverse":
                    {"start_time":[6,0,0],"end_time":[6,5,0], "day":[]},
                "maintainTopDownData":
                    {"start_time":[1,0,0],"end_time":[1,30,0], "day":[]},
                "maintainFactors":
                    {"start_time":[5,50,0],"end_time":[5,56,0], "day":[]}
                },
            "reportingSuite":
                {"createCountrySectorWeights":
                    {"start_time":[9,40,0],"end_time":[9,50,0],"day":[]},
                "calcPerformance":
                    {"start_time":[6,12,0],"end_time":[6,18,0], "day":[]},
                "createVariances":
                    {"start_time":[0,0,1], "end_time":[0,0,6], "day":[0]},
                "createCorrelations":
                    {"start_time":[0,0,10], "end_time":[0,0,20], "day":[0]}
                },
            "optimize":
                {"start_time":[7,40,0],"end_time":[7,46,0], "day":[]},
            "updateWeights":
                {'start_time':[7,50,0],'end_time':[7,55,0], "day":[]},
            "rebalance":
                {"start_time":[9,30,35],"end_time":[9,35,0], "day":[]},
            "deepLearning":
                {'neuralNetworkPreperation':
                    {"start_time":[6,6,0], "end_time":[6,11,0], "day":[]},
                'checkTrainStatus':
                    {"start_time":[6,12,0], "end_time":[6,20,0], 'day':[]},
                'predict':
                    {'start_time':[7,20,0], "end_time":[7,30,0], 'day':[]}
                }
            }

    def getTiming(self,lastUpdate:str, callingFunction:'list[str]') -> bool:

        timingInfo = self.timeRules
        for key in callingFunction:
            timingInfo = timingInfo[key]


        start = timingInfo['start_time']
        end = timingInfo['end_time']
        day = timingInfo['day']

        currDay = date.today().strftime("%Y-%m-%d")
        dayOfWeek = date.today().weekday()
        currTime = datetime.now().time()
        startTime = dt.time(start[0],start[1],start[2])
        endTime = dt.time(end[0], end[1], end[2])

        if startTime <= currTime <= endTime and lastUpdate != currDay and ((len(day) == 0) or (dayOfWeek in day)):
            print("Running " + str(callingFunction))
            return True
        else:
            print("Sleeping " + str(callingFunction))
            return False

