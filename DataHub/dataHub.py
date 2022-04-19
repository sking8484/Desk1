import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataLink import dataLink
from iexLink import iexLink
from privateKeys.privateData import credentials
from pandas.tseries.offsets import *
import datetime as dt
from datetime import datetime, date
from datetime import timedelta
import time

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
        print(fromDate)
        data = self.iexLink.getStockData(currCols,fromDate)
        self.dataLink.append(self.mainStockTable,data)

        colsToAdd = self.positionsToAdd()
        print(colsToAdd)
        if len(colsToAdd) > 0:
            data = self.iexLink.getStockData(colsToAdd, "20050101")
            self.dataLink.joinTables("date",self.mainStockTable,data)

    def timeRules(self, start:list, end:list) -> None:
        updated = None
        while True:
            currDay = date.today().strftime("%Y-%m-%d")
            currTime = datetime.now().time()
            startTime = dt.time(start[0],start[1],start[2])
            endTime = dt.time(end[0], end[1], end[2])
            if np.is_busday(currDay) and startTime <= currTime <= endTime and updated != currDay:
                updated = currDay
                self.updateStockData()
            else:
                print("Sleeping")
                print(currTime)
                print(startTime)
                time.sleep(120)
            


data = dataHub()
data.timeRules([22,0,0],[23,55,0])

