
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
        lastUpdate = ""

        """
        This checks if we are between 6 and 7 am PST, and then gets the data, calculates the weights and uploads to SQL
        Most of this will be abstracted away.
        """
        
        while True:
            if self.TimeRules.rebalanceTimeRules([6,0,0],[7,0,0], lastUpdate):
                DataLink = dataLink(self.credents.credentials)
                data = DataLink.returnTable("test1")
                weights_df = self.ion.getOptimalWeights(data)
                DataLink.append("testModelHoldings",weights_df)
                lastUpdate = date.today().strftime("%Y-%m-%d")
            else:
                print("Sleeping rebalance")
                time.sleep(10)
            


    




controller = ella()
t1 = threading.Thread(target=controller.runDataHub).start()
t2 = threading.Thread(target = controller.rebalance).start()


