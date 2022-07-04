from inspect import trace
from DataHub.dataHub import dataHub
from DataHub.dataLink import dataLink
import pandas as pd
import numpy as np
from datetime import date
from DataHub.privateKeys.privateData import credentials
from datetime import datetime, timedelta
import traceback
import threading
from timeRules import TimeRules
import time

class reportingSuite:
    def __init__(self):
        self.credents = credentials()
        self.TimeRules = TimeRules()

    def calcPerformance(self):
        lastUpdate = ''
        while True:
            if self.TimeRules.getTiming(lastUpdate, ['reportingSuite','calcPerformance']):
                lastUpdate = date.today().strftime("%Y-%m-%d")
                DataLink = dataLink(self.credents.credentials)
                stockData = DataLink.returnTable(self.credents.mainStockTable, pivotObj={'index':'date', 'columns':['symbol'], 'values':['value']})
                modelWeights = DataLink.returnTable(self.credents.weightsTable)

                recentStockWeekday = pd.to_datetime(stockData.iloc[-1,:]['date'][0]).weekday()
                recentDate = pd.to_datetime(stockData.iloc[-1,:]['date'][0])
                daysToShiftBack = 0

                optimizationWeightsAsOf = (recentDate - timedelta(days = daysToShiftBack)).strftime("%Y-%m-%d")

                modelWeightsAsOf = modelWeights[modelWeights['date'] == optimizationWeightsAsOf]

                priceChanges = stockData.drop(columns = 'date').astype(float).replace(to_replace=0,method='ffill').pct_change().iloc[-1,:].reset_index().drop(columns = 'level_0')
                priceChanges.columns = ['symbol', 'value']

                pct_change = 0
                for ticker in modelWeightsAsOf['symbol']:

                    pct_change += modelWeightsAsOf[modelWeightsAsOf['symbol'] == ticker]['value'].astype(float).values[0] * priceChanges[priceChanges['symbol'] == ticker]['value'].values[0]

                data = {"date":[recentDate.strftime("%Y-%m-%d")],'symbol':'pct_change', "value":[pct_change]}
                data = pd.DataFrame.from_dict(data)



                try:
                    currPerfDate = DataLink.getAggElement(self.credents.perfTable, 'date', 'MAX', {'column':'symbol', 'value':'pct_change'})
                    if currPerfDate == recentDate.strftime("%Y-%m-%d"):
                        print("Recent dates perf already recording, skipping perf calcs")
                    else:
                        DataLink.append(self.credents.perfTable,data)
                except Exception as e:
                    print(traceback.print_exc())
                    DataLink.append(self.credents.perfTable,data)
            else:
                time.sleep(self.credents.sleepSeconds)

    def createCountrySectorWeights(self):
        lastUpdate = ""
        while True:
            if self.TimeRules.getTiming(lastUpdate, ['reportingSuite','createCountrySectorWeights']):
                lastUpdate = date.today().strftime("%Y-%m-%d")
                DataLink = dataLink(self.credents.credentials)
                weightsTable = DataLink.returnTable(self.credents.weightsTable).rename(columns = {'value':'weight'})
                csInfoTable = DataLink.returnTable(self.credents.stockInfoTable)
                maxWeightsDate = max(weightsTable['date'])
                maxCSInfoDate = max(csInfoTable['date'])

                currWeights = weightsTable[weightsTable['date'] == maxWeightsDate]

                currCSInfo = csInfoTable[csInfoTable['date'] == maxCSInfoDate]

                csWeights = currWeights.merge(currCSInfo, on='symbol')[['symbol','weight','descriptor','value']]

                csWeights['date'] = lastUpdate
                DataLink.append(self.credents.topDownWeights, csWeights)
            else:
                time.sleep(self.credents.sleepSeconds)

    def maintainData(self) -> None:
        t1 = threading.Thread(target = self.calcPerformance).start()
        t2 = threading.Thread(target = self.createCountrySectorWeights).start()







