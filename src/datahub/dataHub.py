from inspect import trace
from optparse import Values
from threading import Timer
import traceback
import pandas as pd
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
        print(os.environ)
        #self.credents = credentials()
        #self.factors = iexFactors
        #self.token = self.credents.iexToken
        #self.alpacaLink = iexLink(dataLink)
        self.dataLink = dataLink()
        self.mainStockTable = os.environ["MAINSTOCKTABLE"]
        self.mainFactorTable = os.environ["MAINFACTORTABLE"]
        self.alpacaLink = AlpacaLink(dataLink)

    def getBuyUniverse(self, table) -> list:
        if (table == self.mainStockTable):
            payload = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
            stock_table = payload[2]
            universe = list(stock_table['Symbol'].values)
            return universe
        elif (table == self.mainFactorTable):
            return [identifier['symbol'] for identifier in self.factors]

    def updateTimeSeriesData(self, table, universe) -> None:
        #self.removeNonBuyList(table)

        data = self.alpacaLink.get_timeseries_data(universe, table)
        if not data.empty:
            self.dataLink.append(table, data)

    def maintainUniverse(self) -> None:
        try:
            buyUniverse = self.getBuyUniverse(self.mainStockTable)
            self.updateTimeSeriesData(self.mainStockTable, buyUniverse)
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

    '''
    def maintainFactors(self) -> None:
        try:
            self.updateTimeSeriesData(self.mainFactorTable, ['IVV'])
        except Exception as e:
            print(traceback.print_exc())
