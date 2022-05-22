
from concurrent.futures import thread
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
    def __init__(self) -> None:
        self.credents = credentials()
        self.ion = ion()
        self.TimeRules = TimeRules()
        
        

    def runDataHub(self) -> None:
        datahub = dataHub()
        datahub.maintainUniverse()

    
    def optimize(self) -> None:

        lastUpdate = ""
        while True:
            if self.TimeRules.optimizeTimeRules(lastUpdate):
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
    
    def updateWeights(self) -> None:
        lastUpdate = ""

        """
        This checks if we are between 6 and 7 am PST, and then gets the data, calculates the weights and uploads to SQL
        Most of this will be abstracted away.
        """
        
        while True:
            if self.TimeRules.updateWeightsTimeRules(lastUpdate):
                try:
                    lastUpdate = date.today().strftime("%Y-%m-%d")
                    DataLink = dataLink(self.credents.credentials)
                    weights_df = pd.read_csv(self.credents.weightsFile)
                    DataLink.append(self.credents.weightsTable,weights_df)

                except Exception as e:
                    print("The following error occured at " + datetime.now().strftime("%Y-%m-%d-%H-%M"))
                    print(e)
            else:
                print("Sleeping weights update")
                time.sleep(600)

    def rebalance(self) -> None:
        lastUpdate = ""

        """
        This checks if we are between 6 and 7 am PST, and then gets the data, calculates the weights and uploads to SQL
        Most of this will be abstracted away.
        """
        link = alpacaLink()
        while True:
            if self.TimeRules.rebalanceTimeRules(lastUpdate):
                link.rebalance()
                try:
                    lastUpdate = date.today().strftime("%Y-%m-%d")
                    link.rebalance()
                    

                except Exception as e:
                    print("The following error occured at " + datetime.now().strftime("%Y-%m-%d-%H-%M"))
                    print(e)
            else:
                print("Sleeping rebalance")
                time.sleep(600)


    def performanceCalc(self):
        self.ReportingSuite = reportingSuite()
        lastUpdate = ""
        while True:
            if self.TimeRules.performanceTimeRules(lastUpdate):
                try:
                    lastUpdate = date.today().strftime("%Y-%m-%d")
                    self.ReportingSuite.calcPerformance()
                except Exception as e:
                    print("The following error occured at " + datetime.now().strftime("%Y-%m-%d-%H-%M"))
                    print(e)
            else:
                print("Sleeping performance Calc")
                time.sleep(600)
    
            

controller = ella()

optimization = input("If this it the optimization file, please enter -opt: ")
if optimization == "-opt":
    t3 = threading.Thread(target = controller.optimize).start()
else:

    from DataHub.dataHub import dataHub
    from DataHub.dataLink import dataLink
    from reportingSuite.reportingSuite import reportingSuite
    from alpacaLink import alpacaLink

    controller.reportingSuite = reportingSuite()
    t1 = threading.Thread(target = controller.runDataHub).start()
    t2 = threading.Thread(target = controller.rebalance).start()
    t4 = threading.Thread(target = controller.performanceCalc).start()
    t5 = threading.Thread(target = controller.updateWeights).start()



