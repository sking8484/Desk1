
import os 
import sys 
fpath = os.path.join(os.path.dirname(__file__),'DataHub')
sys.path.append(fpath)

from DataHub.dataHub import dataHub
from DataHub.dataLink import dataLink
from ion import ion
from DataHub.privateKeys.privateData import credentials
import threading
import datetime as dt
from datetime import date, datetime
import time
from timeRules import TimeRules

class ella:
    def __init__(self):
        self.credents = credentials()
        self.ion = ion()
        self.TimeRules = TimeRules()
        

    def runDataHub(self):
        datahub = dataHub()
        datahub.maintainUniverse([1,30,0],[2,0,0])

    def rebalance(self):
        i = 0
        DataLink = dataLink(self.credents.credentials)
        while True:
            if self.TimeRules.rebalanceTimeRules(i):
                data = DataLink.returnTable("test1")
                self.ion.getOptimalWeights(data)
                i += 1
            else:
                i += 1
                print("Sleeping rebalance")
                time.sleep(10)
            


    




controller = ella()
t1 = threading.Thread(target=controller.runDataHub).start()
t2 = threading.Thread(target = controller.rebalance).start()


