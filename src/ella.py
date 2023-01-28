
from concurrent.futures import thread
from inspect import trace
import os
import sys
import traceback

fpath = os.path.join(os.path.dirname(__file__),'DataHub')
sys.path.append(fpath)


from ion import ion, orion
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

    def checkTiming(func):
        def wrapper(self, timingId):
            lastUpdate = ""
            while True:
                if self.TimeRules.getTiming(lastUpdate, timingId):
                    func(self)
                    lastUpdate = date.today().strftime("%Y-%m-%d")
                else:
                    time.sleep(self.credents.sleepSeconds)
        return wrapper

    @checkTiming
    def optimize(self) -> None:
        try:
            data = pd.read_csv(self.credents.stockPriceFile).pivot(index = 'date', columns = 'symbol', values = 'value').reset_index()
            predictions = pd.read_csv(self.credents.networkPredictionsLocation)
            weights_df = self.ion.getOptimalWeights(data, delta = 135, leverageAmt=1.9, predictions=predictions, usePredictions=True)
            weights_df.to_csv(self.credents.weightsFile, index = False)

        except Exception as e:
            print(traceback.print_exc())

    @checkTiming
    def updateWeights(self) -> None:


        """
        This checks if we are between 6 and 7 am PST, and then gets the data, calculates the weights and uploads to SQL
        Most of this will be abstracted away.
        """

        try:
            DataLink = dataLink(self.credents.credentials)
            weights_df = pd.read_csv(self.credents.weightsFile)
            predictions_df = pd.read_csv(self.credents.networkPredictionsLocation)
            DataLink.append(self.credents.weightsTable,weights_df)
            DataLink.append(self.credents.networkPredictionsTable, predictions_df)

        except Exception as e:
            print("The following error occured at " + datetime.now().strftime("%Y-%m-%d-%H-%M"))
            print(traceback.print_exc())

    @checkTiming
    def rebalance(self) -> None:

        """
        This checks if we are between 6 and 7 am PST, and then gets the data, calculates the weights and uploads to SQL
        Most of this will be abstracted away.
        """
        try:
            threadRebalance()
        except Exception as e:
            print(traceback.print_exc())


    def runReportingSuite(self):
        self.ReportingSuite = reportingSuite()
        self.ReportingSuite.maintainData()

    @checkTiming
    def neuralNetworkPreperation(self):

        try:
            DataLink = dataLink(self.credents.credentials)
            prices = DataLink.returnTable(self.credents.mainStockTable)
            factors = DataLink.returnTable(self.credents.mainFactorTable)

            prices.to_csv(self.credents.networkPricesLocation, index = False)
            factors.to_csv(self.credents.networkFactorsLocation, index = False)
        except Exception as e:
            print(traceback.print_exc())


    def neuralNetwork(self):
        orionObj = orion()
        orionObj.controlNetwork()

controller = ella()

options = {
    '-opt':"Optimization",
    '-hub':'Main Script',
    '-net':"Neural Network"
}


selectionString = ''
for key in options:
    selectionString += f'{key}: {options[key]}\n'


programSelector = input("Please select which program to run \n" + selectionString)
if programSelector == "-opt":
    t3 = threading.Thread(target = controller.optimize, args = (['optimize'],)).start()
elif programSelector == '-net':
    t7 = threading.Thread(target = controller.neuralNetwork).start()
elif programSelector == '-hub':

    from DataHub.dataHub import dataHub
    from DataHub.dataLink import dataLink
    from reportingSuite.reportingSuite import reportingSuite
    from alpacaLink import threadRebalance

    controller.reportingSuite = reportingSuite()
    t1 = threading.Thread(target = controller.runDataHub).start()
    t2 = threading.Thread(target = controller.rebalance, args = (['rebalance'],)).start()
    t4 = threading.Thread(target = controller.runReportingSuite).start()
    t5 = threading.Thread(target = controller.updateWeights, args = (['updateWeights'],)).start()
    t6 = threading.Thread(target = controller.neuralNetworkPreperation, args = (['deepLearning', 'neuralNetworkPreperation'],)).start()



