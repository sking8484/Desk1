from inspect import trace
from DataHub.dataHub import dataHub
from DataHub.dataLink import dataLink
import pandas as pd
import numpy as np
from datetime import date
from DataHub.privateKeys.privateData import credentials
from datetime import datetime, timedelta
import traceback


class reportingSuite:
    def __init__(self):
        self.credents = credentials()

    def calcPerformance(self):
        DataLink = dataLink(self.credents.credentials)
        stockData = DataLink.returnTable(self.credents.mainStockTable, pivotObj={'index':'date', 'columns':['symbol'], 'values':['value']})
        modelWeights = DataLink.returnTable(self.credents.weightsTable)

        recentStockWeekday = pd.to_datetime(stockData.iloc[-1,:]['date'][0]).weekday()
        recentDate = pd.to_datetime(stockData.iloc[-1,:]['date'][0])
        daysToShiftBack = 0

        optimizationWeightsAsOf = (recentDate - timedelta(days = daysToShiftBack)).strftime("%Y-%m-%d")

        modelWeightsAsOf = modelWeights[modelWeights['date'] == optimizationWeightsAsOf]

        priceChanges = stockData.drop(columns = 'date').astype(float).replace(to_replace=0,method='ffill').pct_change().iloc[-1,:]
        pct_change = 0
        for ticker in modelWeightsAsOf['symbol']:
            pct_change += modelWeightsAsOf[modelWeightsAsOf['symbol'] == ticker]['value'].astype(float).values[0] * priceChanges[ticker]

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





