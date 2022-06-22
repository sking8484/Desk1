
from concurrent.futures import thread
from inspect import trace
import os
import sys
import traceback
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
        datahub.maintainData()


    def optimize(self) -> None:

        lastUpdate = ""
        while True:
            if self.TimeRules.getTiming(lastUpdate, ['optimize']):
                try:
                    lastUpdate = date.today().strftime("%Y-%m-%d")
                    data = pd.read_csv(self.credents.stockPriceFile).pivot(index = 'date', columns = 'symbol', values = 'value').reset_index()
                    print(data)
                    weights_df = self.ion.getOptimalWeights(data, delta = 65, leverageAmt=1.98)
                    weights_df.to_csv(self.credents.weightsFile, index = False)

                except Exception as e:
                    print(traceback.print_exc())
            else:
                time.sleep(self.credents.sleepSeconds)

    def updateWeights(self) -> None:
        lastUpdate = ""

        """
        This checks if we are between 6 and 7 am PST, and then gets the data, calculates the weights and uploads to SQL
        Most of this will be abstracted away.
        """

        while True:
            if self.TimeRules.getTiming(lastUpdate, ['updateWeights']):
                try:
                    lastUpdate = date.today().strftime("%Y-%m-%d")
                    DataLink = dataLink(self.credents.credentials)
                    weights_df = pd.read_csv(self.credents.weightsFile)
                    DataLink.append(self.credents.weightsTable,weights_df)

                except Exception as e:
                    print("The following error occured at " + datetime.now().strftime("%Y-%m-%d-%H-%M"))
                    print(traceback.print_exc())

            else:
                time.sleep(self.credents.sleepSeconds)

    def rebalance(self) -> None:
        lastUpdate = ""

        """
        This checks if we are between 6 and 7 am PST, and then gets the data, calculates the weights and uploads to SQL
        Most of this will be abstracted away.
        """
        link = alpacaLink()
        while True:
            if self.TimeRules.getTiming(lastUpdate, ['rebalance']):
                try:
                    lastUpdate = date.today().strftime("%Y-%m-%d")
                    link.rebalance()
                except Exception as e:
                    print(traceback.print_exc())
            else:
                time.sleep(self.credents.sleepSeconds)

    def performanceCalc(self):
        self.ReportingSuite = reportingSuite()
        lastUpdate = ""
        while True:
            if self.TimeRules.getTiming(lastUpdate, ['performanceCalc']):
                try:
                    lastUpdate = date.today().strftime("%Y-%m-%d")
                    self.ReportingSuite.calcPerformance()
                except Exception as e:

                    print(traceback.print_exc())
            else:

                time.sleep(self.credents.sleepSeconds)
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



