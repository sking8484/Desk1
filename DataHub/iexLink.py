from privateKeys.privateData import credentials
from dataLink import dataLink
import pandas as pd
from datetime import date
import requests
from datetime import datetime
import time

"""
Grab last item in stock table in order to get the date
get list of business days using pandas
at each business day, append to stock table, and then append stock table to larger stock table.
then upload this to sql
"""

class iexLink:
    def __init__(self):
        credents = credentials()
        self.token = credents.iexToken
        #self.dataLink = dataLink(credents.credentials)

    def getStockData(self, tickers: list, startDate: str, sinceSpecificDate = True) -> pd.DataFrame:

        firstJoin = True
        BaseUrl = "https://cloud.iexapis.com/"
        version = "stable/"
        token = "&token=" + self.token

        for stock in tickers:
            time.sleep(.1)
            myParams = "time-series/HISTORICAL_PRICES/" + stock + "?from=" + startDate + "&to=" + date.today().strftime("%Y%m%d")
            base_url = BaseUrl + version + myParams + token

            data = requests.get(base_url)

            stockData = pd.DataFrame.from_dict(data.json(), orient="columns")[['date','close']]
            stockData.columns = ['date', stock]

            if firstJoin == True:
                historicalData = stockData
                firstJoin = False
            else:
                historicalData = historicalData.merge(stockData, on = 'date', how = 'left')

        historicalData['date'] = pd.to_datetime(historicalData['date'], unit = 'ms')
        historicalData = historicalData.sort_values(by = "date")
        historicalData['date'] = historicalData['date'].dt.strftime("%Y-%m-%d")

        return historicalData

    def countrySectorInfo(self, tickers: list) -> pd.DataFrame:
        firstJoin = True
        BaseUrl = "https://cloud.iexapis.com/"
        version = "stable/"
        token = "token=" + self.token

        for stock in tickers:
            #time.sleep(.1)
            myParams = 'stock/' + stock + '/company?'
            base_url = BaseUrl + version + myParams + token

            data = requests.get(base_url)

            stockData = pd.DataFrame.from_dict([data.json()], orient="columns")[['symbol', 'sector', 'industry', 'country']]
            stockData.columns = ['symbol', 'sector','industry','country']

            if firstJoin == True:
                historicalData = stockData
                firstJoin = False
            else:
                historicalData = pd.concat([historicalData,stockData], ignore_index=True)

        historicalData['date'] = datetime.today().strftime('%Y-%m-%d')

        return historicalData

    def getFactorData(self) -> None:
        # Get factor definition information from iexFactors.py
        pass





