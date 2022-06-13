from inspect import trace
from optparse import Values
from threading import Timer
import traceback
from turtle import towards
import pandas as pd
from dataLink import dataLink
from iexLink import iexLink
from privateKeys.privateData import credentials
from pandas.tseries.offsets import *
import datetime as dt
from datetime import datetime, date
from datetime import timedelta
import time
import threading
from timeRules import TimeRules
from iexDefinitions import iexFactors

class dataHub:
    def __init__(self):
        self.TimeRules = TimeRules()
        self.credents = credentials()
        self.factors = iexFactors
        self.token = self.credents.iexToken
        self.iexLink = iexLink()
        self.mainStockTable = self.credents.mainStockTable
        self.mainFactorTable = self.credents.mainFactorTable

    def getBuyUniverse(self, table) -> list:
        if (table == self.mainStockTable):
            payload = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
            stock_table = payload[2]
            universe = list(stock_table['Symbol'].values)
            return universe
        elif (table == self.mainFactorTable):
            return [identifier['symbol'] for identifier in self.factors]

    def createTickerObject(self, ticker):
        '''
        {
            "symbol":"CPI",
            "timeSeriesUrlParam":"/economic/CPIAUCSL",
            "frequency":"M",
            "columnsToKeep":['date','value'],
            "columnNames":['date','CPI']
        }
        '''

        return {
            "symbol":ticker,
            "timeSeriesUrlParam":"HISTORICAL_PRICES/" + ticker,
            "frequency":"D",
            "columnsToKeep":['date','close'],
            "columnNames":['date','value'],
            'tableName':self.mainStockTable
        }

    def updateTimeSeriesData(self, table) -> None:
        #self.removeNonBuyList(table)
        self.buyUniverse = self.getBuyUniverse(table)

        if (table == self.mainStockTable):
            currIdentifiers = [self.createTickerObject(col) for col in self.buyUniverse]

        elif (table == self.mainFactorTable):
            currIdentifiers = [identifier for identifier in self.factors if identifier['symbol'] in  self.buyUniverse]

        data = self.iexLink.getTimeSeriesData(currIdentifiers)
        self.dataLink.append(table, data)


    def maintainUniverse(self) -> None:
        lastUpdate = ""

        while True:
            if self.TimeRules.getTiming(lastUpdate, ['dataHub', 'maintainUniverse']):
                self.dataLink = dataLink(self.credents.credentials)
                lastUpdate = date.today().strftime("%Y-%m-%d")
                try:
                    self.updateTimeSeriesData(self.mainStockTable)
                except Exception as e:
                    print(traceback.print_exc())

                data = self.dataLink.returnTable(self.mainStockTable)
                data.to_csv(self.credents.stockPriceFile,index=False)
            else:
                time.sleep(self.credents.sleepSeconds)

    def maintainTopDownData(self) -> None:

        lastUpdate = ""

        while True:
            if self.TimeRules.getTiming(lastUpdate, ['dataHub', 'maintainTopDownData']):
                lastUpdate = date.today().strftime("%Y-%m-%d")
                self.dataLink = dataLink(self.credents.credentials)
                topDownData = self.iexLink.countrySectorInfo(self.getBuyUniverse(self.mainStockTable))

                try:
                    self.dataLink.append(self.credents.stockInfoTable, topDownData)
                except Exception as e:
                    print(traceback.print_exc())
                self.dataLink.closeConnection()
            else:
                time.sleep(self.credents.sleepSeconds)

    def maintainFactors(self) -> None:
        lastUpdate = ""

        while True:
            if self.TimeRules.getTiming(lastUpdate, ['dataHub', 'maintainFactors']):
                lastUpdate = date.today().strftime("%Y-%m-%d")
                self.dataLink = dataLink(self.credents.credentials)
                try:
                    self.updateTimeSeriesData(self.mainFactorTable)
                except Exception as e:
                    print(traceback.print_exc())
                self.dataLink.closeConnection()
            else:
                time.sleep(self.credents.sleepSeconds)


    def maintainData(self) -> None:
        t1 = threading.Thread(target = self.maintainUniverse).start()
        t2 = threading.Thread(target = self.maintainTopDownData).start()
        t3 = threading.Thread(target = self.maintainFactors).start()



