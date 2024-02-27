from inspect import trace
from optparse import Values
from threading import Timer
import traceback
import pandas as pd
from privateKeys.privateData import credentials
from pandas.tseries.offsets import *
import datetime as dt
from datetime import datetime, date
from datetime import timedelta
import time
import threading
import os
from dotenv import load_dotenv
from datahub.alpacaLink import AlpacaLink

class dataHub:
    def __init__(self, dataLink):
        load_dotenv()
        #self.credents = credentials()
        #self.factors = iexFactors
        #self.token = self.credents.iexToken
        #self.alpacaLink = iexLink(dataLink)
        self.mainStockTable = "TEST_STOCK_TABLE"
        self.mainFactorTable = "BLEH"
        self.dataLink = dataLink()
        self.alpacaLink = AlpacaLink(dataLink)

    def getBuyUniverse(self, table) -> list:
        if (table == self.mainStockTable):
            payload = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
            stock_table = payload[2]
            universe = list(stock_table['Symbol'].values)
            return universe
        elif (table == self.mainFactorTable):
            return [identifier['symbol'] for identifier in self.factors]

    def updateTimeSeriesData(self, table) -> None:
        #self.removeNonBuyList(table)
        self.buyUniverse = self.getBuyUniverse(table)

        data = self.alpacaLink.get_timeseries_data(self.buyUniverse)
        self.dataLink.append(table, data)


    def maintainUniverse(self) -> None:
        try:
            self.updateTimeSeriesData(self.mainStockTable)
        except Exception as e:
            print(traceback.print_exc())
    '''
    def maintainTopDownData(self) -> None:

        self.dataLink = dataLink(self.credents.credentials)
        topDownData = self.alpacaLink.countrySectorInfo(self.getBuyUniverse(self.mainStockTable))

        try:
            self.dataLink.append(self.credents.stockInfoTable, topDownData)
        except Exception as e:
            print(traceback.print_exc())
        self.dataLink.closeConnection()

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

    '''


    def maintainData(self) -> None:
        t1 = threading.Thread(target = self.maintainUniverse).start()
        t2 = threading.Thread(target = self.maintainTopDownData).start()



