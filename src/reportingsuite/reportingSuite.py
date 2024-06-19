from inspect import trace
import sys
import os
from db_link.db_link import DataLink
import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime, timedelta
import traceback
import threading
import time
import os
from dotenv import load_dotenv



class reportingSuite:
    def __init__(self):
        load_dotenv()

    def getFromDate(self, identifier:dict):
        db_link = DataLink()
        try:
            date = db_link.get_agg_element(identifier['tableName'], 'date', 'MAX', {'column':'symbol', 'value':identifier['symbol']})
        except Exception:
            date = '2005-01-01'
        if date == None:
            return '2005-01-01'
        fromDate = pd.to_datetime(date)
        return pd.to_datetime(fromDate)


    def switchTimesToDates(self, dates):
        new_dates = {}
        for date in dates:
            new_dates[pd.to_datetime(date.strftime("%Y-%m-%d"))] =  max(new_dates.get(pd.to_datetime(date.strftime("%Y-%m-%d")), pd.to_datetime(date)), pd.to_datetime(date))
        new_dates_switched = {}
        for k, v in new_dates.items():
            new_dates_switched[v] = k

        returnDates = []
        for date in dates:
            returnDates.append(new_dates_switched.get(date, date))

        return returnDates

    def calcPerformance(self):

        '''
        {
            "symbol":"pct_change",
            'tableName':credents.perfTable
        },

        '''


        link = DataLink()
        identifier = {
            "symbol":"pct_change",
            'tableName':os.environ["MAINPERFTABLE"]
        }
        maxPerfDate = self.getFromDate(identifier)
        print(maxPerfDate)
        stockData = link.return_table(os.environ["MAINSTOCKTABLE"]).pivot(index = "date", columns = "symbol", values = "value").rename_axis(columns=None)
        stockData = stockData.astype(float).pct_change().fillna(0.0)
        newDates = self.switchTimesToDates(pd.to_datetime(stockData.index))
        stockData.index = newDates
        #stockData.set_index(self.switchTimesToDates(pd.to_datetime(stockData.index)), inplace=True)
        stockData = (stockData.loc[maxPerfDate:]).iloc[1:]
        modelWeights = link.return_table(os.environ["MAINWEIGHTSTABLE"]).pivot(index = "date", columns = "symbol", values = "value").rename_axis(columns=None).astype(float).fillna(0.0)
        newDates = self.switchTimesToDates(pd.to_datetime(modelWeights.index))
        modelWeights.index = newDates
        modelWeights = modelWeights[modelWeights.index.isin(stockData.index)]
        stockData = stockData[stockData.index.isin(modelWeights.index)]
        print(modelWeights)



        modelWeights = modelWeights[stockData.columns]
        returns = modelWeights @ stockData.T
        data = pd.DataFrame(data = np.diag(returns), index = returns.index)
        data.reset_index(inplace= True)
        data.rename(columns= {0:'value', 'index':'date'}, inplace=True)
        data['symbol'] = 'pct_change'

        data['date'] = pd.to_datetime(data['date'])

        try:
            link.append(os.environ["MAINPERFTABLE"],data)
        except Exception as e:
            print(traceback.print_exc())
            link.append(os.environ["MAINPERFTABLE"],data)
