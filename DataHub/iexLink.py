from csv import excel_tab
from os import times
from privateKeys.privateData import credentials
from dataLink import dataLink
import pandas as pd
from datetime import date
import requests
from datetime import datetime
import time
from iexDefinitions import iexFactors
from pandas.tseries.offsets import *
from dataLink import dataLink

"""
Grab last item in stock table in order to get the date
get list of business days using pandas
at each business day, append to stock table, and then append stock table to larger stock table.
then upload this to sql
"""

class iexLink:
    def __init__(self):
        self.credents = credentials()
        self.token = self.credents.iexToken
        self.iexFactors = iexFactors
        self.BaseUrl = self.credents.iexBaseUrl
        self.version = "stable/"
        #self.dataLink = dataLink(credents.credentials)

    def getFromDate(self, identifier:dict):
        try:
            date = self.dataLink.getAggElement(identifier['tableName'], 'date', 'MAX', {'column':'symbol', 'value':identifier['symbol']})
        except Exception:
            date = '2005-01-01'
        if date == None:
            return '2005-01-01'
        fromDate = (pd.to_datetime(date) + BusinessDay()).strftime("%Y-%m-%d")
        return fromDate

    def getTimeSeriesData(self, identifiers: 'list[dict]') -> pd.DataFrame:
        self.dataLink = dataLink(self.credents.credentials)
        '''
        {
            "symbol":"CPI",
            "timeSeriesUrlParam":"/economic/CPIAUCSL",
            "frequency":"M",
            "columnsToKeep":['date','value'],
            "columnNames":['date','CPI']
        }
        '''
        historicalData = ""
        firstJoin = True
        for identifier in identifiers:
            print(identifier)
            time.sleep(.1)
            myParams = "time-series/" + identifier['timeSeriesUrlParam'] + "?from=" + self.getFromDate(identifier) + "&to=" + date.today().strftime("%Y%m%d")
            requestUrl = self.BaseUrl + self.version + myParams +"&token="+ self.token
            data = requests.get(requestUrl)
            try:
                timeSeriesData = pd.DataFrame.from_dict(data.json(), orient="columns")[identifier['columnsToKeep']]
                timeSeriesData.columns = identifier['columnNames']
            except Exception as e:
                continue

            timeSeriesData['symbol'] = identifier['symbol']
            timeSeriesData['date'] = pd.to_datetime(timeSeriesData['date'], unit = 'ms')
            timeSeriesData = timeSeriesData.sort_values(by = "date")
            timeSeriesData.set_index('date', inplace=True)
            timeSeriesData = timeSeriesData.resample('B').ffill()
            timeSeriesData.reset_index(inplace=True)
            timeSeriesData['date'] = timeSeriesData['date'].dt.strftime("%Y-%m-%d")

            if firstJoin == True:
                historicalData = timeSeriesData
                firstJoin = False
            else:
                historicalData = pd.concat([historicalData,timeSeriesData])

        self.dataLink.closeConnection()
        return historicalData

    def countrySectorInfo(self, tickers: list) -> pd.DataFrame:
        self.dataLink = dataLink(self.credents.credentials)
        firstJoin = True
        for stock in tickers:
            time.sleep(.1)
            myParams = 'stock/' + stock + '/company?'
            base_url = self.BaseUrl + self.version + myParams +"&token=" + self.token

            data = requests.get(base_url)
            print(data.json())
            stockData = pd.melt(pd.DataFrame.from_dict([data.json()], orient="columns"), id_vars=['symbol'],
                                                                                         value_vars=['sector', 'industry', 'country'],
                                                                                         var_name='descriptor',
                                                                                         value_name='value')

            if firstJoin == True:
                historicalData = stockData
                firstJoin = False
            else:
                historicalData = pd.concat([historicalData,stockData], ignore_index=True)

        historicalData['date'] = datetime.today().strftime('%Y-%m-%d')
        self.dataLink.closeConnection()

        return historicalData







