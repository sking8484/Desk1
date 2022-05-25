from threading import Timer
import pandas as pd
from dataLink import dataLink
from iexLink import iexLink
from privateKeys.privateData import credentials
from pandas.tseries.offsets import *
import datetime as dt
from datetime import datetime, date
from datetime import timedelta
import time
from timeRules import TimeRules

class dataHub:
    def __init__(self):
        self.TimeRules = TimeRules()
        self.credents = credentials()
        self.token = self.credents.iexToken
        self.dataLink = dataLink(self.credents.credentials)
        self.iexLink = iexLink()
        self.mainStockTable = self.credents.mainStockTable
    
    def getBuyUniverse(self) -> list:

        payload=pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
        stock_table = payload[2]
        universe = list(stock_table['Symbol'].values)
        

        return universe 

    def getCurrentUniverse(self) -> list:

        columns = self.dataLink.getLastRow(self.mainStockTable).columns
        return [column for column in columns if column != "ID" and column != "date" and '.' not in column]
    
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
        
        if len(colsToAdd) > 0:
            data = self.iexLink.getStockData(colsToAdd, "20050101")
            self.dataLink.joinTables("date",self.mainStockTable,data)

    def maintainUniverse(self) -> None:
        lastUpdate = ""
        
        while True:
            if self.TimeRules.universeTimeRules(lastUpdate):
                
                self.dataLink = dataLink(self.credents.credentials)
                lastUpdate = date.today().strftime("%Y-%m-%d")
                try:
                    self.updateStockData()
                    
                except Exception as e:
                    print("Update Stock Data threw an exception at " + datetime.now().strftime("%Y-%m-%d-%H-%M"))
                    print(e)
                    
                data = self.dataLink.returnTable(self.mainStockTable)
                data.to_csv(self.credents.stockPriceFile,index=False)
            else:
                print("Universe Sleeping " + datetime.now().strftime("%Y-%m-%d-%H-%M"))
                time.sleep(600)
            



