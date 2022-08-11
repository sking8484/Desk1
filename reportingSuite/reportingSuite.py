from inspect import trace
from tkinter import E
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

    def getFromDate(self, identifier:dict):
        DataLink = dataLink(self.credents.credentials)
        try:
            date = DataLink.getAggElement(identifier['tableName'], 'date', 'MAX', {'column':'symbol', 'value':identifier['symbol']})
        except Exception:
            date = '2005-01-01'
        if date == None:
            return '2005-01-01'
        fromDate = (pd.to_datetime(date)).strftime("%Y-%m-%d")
        return fromDate



    def calcPerformance(self):

        '''
        {
            "symbol":"pct_change",
            'tableName':credents.perfTable
        },

        '''


        lastUpdate = ''
        while True:
            if self.TimeRules.getTiming(lastUpdate, ['reportingSuite','calcPerformance']):
                lastUpdate = date.today().strftime("%Y-%m-%d")
                DataLink = dataLink(self.credents.credentials)
                identifier = {
                    "symbol":"pct_change",
                    'tableName':self.credents.perfTable
                }
                maxPerfDate = self.getFromDate(identifier)
                stockData = DataLink.returnTable(self.credents.mainStockTable, pivotObj={'index':'date', 'columns':['symbol'], 'values':['value']}).set_index('date').astype(float).pct_change().fillna(0.0)
                stockData.set_index(pd.to_datetime(stockData.index), inplace=True)
                stockData = (stockData.loc[maxPerfDate:]).iloc[1:]
                modelWeights = DataLink.returnTable(self.credents.weightsTable, pivotObj = {'index':'date','columns':['symbol'], 'values':['value']}).set_index('date').astype(float).fillna(0.0)
                modelWeights.set_index(pd.to_datetime(modelWeights.index),inplace=True)
                modelWeights = modelWeights.loc[stockData.index]



                modelWeights = modelWeights[stockData.columns]
                returns = modelWeights @ stockData.T
                data = pd.DataFrame(data = np.diag(returns), index = returns.index)
                data.reset_index(inplace= True)
                data.rename(columns= {0:'value'}, inplace=True)
                data['symbol'] = 'pct_change'







                try:
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

    def createVariances(self):
        lastUpdate = ''

        while True:
            if self.TimeRules.getTiming(lastUpdate, ['reportingSuite', 'createVariances']):
                lastUpdate = date.today().strftime("%Y-%m-%d")
                DataLink = dataLink(self.credents.credentials)
                currTable = DataLink.returnTable(self.credents.varianceTable)
                if currTable['date'][0] == lastUpdate:
                    print("Cov table already update")
                else:
                    returnsTable = DataLink.returnTable(self.credents.mainStockTable)
                    returnsTable = returnsTable.pivot(index = 'date', columns = 'symbol', values = "value")
                    covs = returnsTable.iloc[-253:,:].astype(float).pct_change().iloc[-252:,:].var().reset_index().rename(columns = {0:'value'})

                    covs['date'] = lastUpdate
                    print(covs)
                    DataLink.append(self.credents.varianceTable, covs)

            else:
                time.sleep(self.credents.sleepSeconds)

    def createCorrelations(self):
        lastUpdate = ''

        while True:
            if self.TimeRules.getTiming(lastUpdate, ['reportingSuite', 'createCorrelations']):
                lastUpdate = date.today().strftime("%Y-%m-%d")
                DataLink = dataLink(self.credents.credentials)
                currTable = DataLink.returnTable(self.credents.corrTable)
                needsUpdate = True
                try:
                    if currTable['date'][0] == lastUpdate:
                        print("Cov table already update")
                        needsUpdate = False
                    else:
                        pass
                except Exception as e:
                    print(e)

                if needsUpdate:
                    returnsTable = DataLink.returnTable(self.credents.mainStockTable)
                    returnsTable = returnsTable.pivot(index = 'date', columns = 'symbol', values = "value")
                    corrs = returnsTable.iloc[-253:,:].astype(float).pct_change().iloc[-252:,:].corr().melt(ignore_index=False).rename(columns = {'symbol':'symbol2'}).reset_index()

                    corrs['date'] = lastUpdate

                    DataLink.append(self.credents.corrTable, corrs)

            else:
                time.sleep(self.credents.sleepSeconds)

    def maintainData(self) -> None:
        t1 = threading.Thread(target = self.calcPerformance).start()
        t2 = threading.Thread(target = self.createCountrySectorWeights).start()
        t3 = threading.Thread(target = self.createVariances).start()
        t4 = threading.Thread(target = self.createCorrelations).start()







