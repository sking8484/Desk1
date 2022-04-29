
import os 
import sys 
fpath = os.path.join(os.path.dirname(__file__),'DataHub')
sys.path.append(fpath)


from ion import ion
from DataHub.privateKeys.privateData import credentials
import threading
import datetime as dt
from datetime import date, datetime
import time
from timeRules import TimeRules
import pandas as pd

class ella:
    def __init__(self):
        self.credents = credentials()
        self.ion = ion()
        self.TimeRules = TimeRules()
        

    def runDataHub(self):
        datahub = dataHub()
        datahub.maintainUniverse([2,0,0],[2,30,0])

    
    def optimize(self):

        lastUpdate = ""
        while True:
            if self.TimeRules.rebalanceTimeRules([3,0,0],[4,0,0], lastUpdate):
                try:
                    lastUpdate = date.today().strftime("%Y-%m-%d")
                    data = pd.read_csv(self.credents.stockPriceFile)
                    weights_df = self.ion.getOptimalWeights(data)
                    weights_df.to_csv(self.credents.weightsFile, index = False)

                except Exception as e:
                    print("The following error occured at " + datetime.now().strftime("%Y-%m-%d-%H-%M"))
                    print(e)
            else:
                print("Sleeping optimization")
                time.sleep(600)
    
    def rebalance(self):
        lastUpdate = ""

        """
        This checks if we are between 6 and 7 am PST, and then gets the data, calculates the weights and uploads to SQL
        Most of this will be abstracted away.
        """
        
        while True:
            if self.TimeRules.rebalanceTimeRules([4,30,0],[5,0,0], lastUpdate):
                try:
                    lastUpdate = date.today().strftime("%Y-%m-%d")
                    DataLink = dataLink(self.credents.credentials)
                    weights_df = pd.read_csv(self.credents.weightsFile)
                    DataLink.append("testModelHoldings",weights_df)

                except Exception as e:
                    print("The following error occured at " + datetime.now().strftime("%Y-%m-%d-%H-%M"))
                    print(e)
            else:
                print("Sleeping rebalance")
                time.sleep(600)
            

controller = ella()

optimization = input("If this it the optimization file, please enter -opt: ")
if optimization == "-opt":
    t3 = threading.Thread(target = controller.optimize).start()
else:

    from DataHub.dataHub import dataHub
    from DataHub.dataLink import dataLink

    t1 = threading.Thread(target=controller.runDataHub).start()
    t2 = threading.Thread(target = controller.rebalance).start()



