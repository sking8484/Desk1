import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataLink import dataLink
from iexLink import iexLink
from privateKeys.privateData import credentials
from pandas.tseries.offsets import *
import datetime
from datetime import datetime

class dataHub:
    def __init__(self):
        credents = credentials()
        self.token = credents.iexToken
        self.dataLink = dataLink(credents.credentials)
        self.iexLink = iexLink()
        self.mainStockTable = credents.mainStockTable
    
    def timeRules(self):
        pass 
    
    def getBuyUniverse(self) -> list:

        payload=pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
        stock_table = payload[2]
        universe = list(stock_table['Symbol'].values)
        

        return universe

    def getCurrentUniverse(self) -> list:

        columns = self.dataLink.getLastRow(self.mainStockTable).columns
        return [column.replace("_",".") for column in columns if column != "ID" and column != "date"]
    
    def positionsToRemove(self) -> list:

        currUniverse = self.getCurrentUniverse()
        buyUniverse = self.getBuyUniverse()
        posToSell = list(set(currUniverse) - set(buyUniverse))
        return posToSell
    
    def positionsToAdd(self) -> list:
        return list(set(self.getBuyUniverse())- set(self.getCurrentUniverse()))

    def removeNonBuyList(self) -> None:
        posToSell = self.positionsToRemove()
        self.dataLink.dropColumns(self.mainStockTable,posToSell)

    

    
    def updateStockData(self) -> None:
        self.removeNonBuyList()
        currCols = [col.replace("_",".") for col in self.dataLink.getColumns(self.mainStockTable)  if col != "ID" and col != "date"]
        fromDate = (pd.to_datetime(self.dataLink.getLastRow(self.mainStockTable)['date'][0]) + BusinessDay()).strftime("%Y-%m-%d")
        data = self.iexLink.getStockData(currCols,fromDate)
        self.dataLink.append(self.mainStockTable,data)

        colsToAdd = self.positionsToAdd()
        print(colsToAdd)
        if len(colsToAdd) > 0:
            data = self.iexLink.getStockData(colsToAdd, "20050101")
            self.dataLink.joinTables("date","testStockTable",data)

        



data = dataHub()
data.updateStockData()