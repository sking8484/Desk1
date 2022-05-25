from DataHub.dataHub import dataHub
from DataHub.dataLink import dataLink
import pandas as pd
import numpy as np
from datetime import date 
from DataHub.privateKeys.privateData import credentials 
from datetime import datetime, timedelta


class reportingSuite:
    def __init__(self):
        self.credents = credentials() 

    def calcPerformance(self):
        DataLink = dataLink(self.credents.credentials)
        stockData = DataLink.returnTable(self.credents.mainStockTable)
        modelWeights = DataLink.returnTable(self.credents.weightsTable)

        recentStockWeekday = pd.to_datetime(stockData.iloc[-1,:]['date']).weekday()
        recentDate = pd.to_datetime(stockData.iloc[-1,:]['date'])
        daysToShiftBack = recentStockWeekday + 3

        optimizationWeightsAsOf = (recentDate - timedelta(days = daysToShiftBack)).strftime("%Y-%m-%d")
        modelWeightsAsOf = modelWeights[modelWeights['date'] == optimizationWeightsAsOf]
        
        priceChanges = stockData.drop(columns = 'date').astype(float).replace(to_replace=0,method='ffill').pct_change().iloc[-1,:]
        pct_change = 0
        for ticker in modelWeightsAsOf['Ticker']:
            pct_change += modelWeightsAsOf[modelWeightsAsOf['Ticker'] == ticker]['weights'].astype(float).values[0] * priceChanges[ticker]
            
        data = {"date":[recentDate.strftime("%Y-%m-%d")],"pct_change":[pct_change]}
        data = pd.DataFrame.from_dict(data)
        
        currPerf = DataLink.returnTable(self.credents.perfTable)
        
        if currPerf.iloc[-1,:]['date'] == recentDate.strftime("%Y-%m-%d"):
            print("Recent dates perf already recording, skipping perf calcs")
        else:    
            try:
                DataLink.append(self.credents.perfTable,data)
            except Exception as e:
                print("Couldn't Append. Trying to create")
                try:
                    DataLink.createTable(self.credents.perfTable, data)
                except Exception as e:
                    print(e)
                    print("Couldn't create table either. Needs to be fixed")
            
        

        
        
