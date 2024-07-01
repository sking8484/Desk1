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

    def calcStats(self):
        link = DataLink()

        perfData = self.buildRelevantData(link)
        lastRow = perfData.iloc[-1]
        returns_vs_index = (lastRow['cumulative'] - lastRow['IVV'])*100
        sharpeRatio = self.calculateSharpeRatio(perfData)
        dailyRisk = perfData['pct_change'].std()*100

        num_positions = self.get_num_positions(link)

        data_dict = {'date':[datetime.now()], 'sharpe_ratio':[sharpeRatio], 'returns_vs_index':[returns_vs_index], 'num_positions':[num_positions], 'mean_daily_risk':[dailyRisk]}
        df = pd.DataFrame.from_dict(data_dict)
        melted = pd.melt(df, id_vars = ['date'], var_name = 'symbol')

        link.append(os.environ["MAINSTATSTABLE"], melted)

    def get_num_positions(self, link):
        positionsData = link.return_table(os.environ["MAINWEIGHTSTABLE"]).set_index('date')[['value']].astype(float)
        positionsDataFiltered = positionsData[positionsData['value'] > 0].loc[max(positionsData.index.values)]
        return len(positionsDataFiltered['value'].values)


    def calculateSharpeRatio(self, perfData):
        mean_daily_returns = perfData['pct_change'].mean()
        annualized_daily_returns = (1+mean_daily_returns)**365 - 1
        
        mean_daily_risk = perfData['pct_change'].std()
        annualized_daily_risk = (1+mean_daily_risk)**365 - 1

        return (annualized_daily_returns - .03)/annualized_daily_risk




    def buildRelevantData(self, link):
        perfData = link.return_table(os.environ["MAINPERFTABLE"]).pivot(index = "date", columns = "symbol", values = "value").rename_axis(columns=None).astype(float)
        perfDataIndex = pd.to_datetime(perfData.index).date
        perfData.index = perfDataIndex

        perfDataRollingYear = perfData.iloc[-252:]
        perfDataRollingYear['cumulative'] = ((1+perfDataRollingYear['pct_change']).cumprod())
        perfDataRollingYear = perfDataRollingYear[['cumulative']]

        sp500Data = link.return_table(os.environ["MAINFACTORTABLE"]).pivot(index = "date", columns = "symbol", values = "value").rename_axis(columns=None).astype(float)[["IVV"]]
        sp500DataIndex = pd.to_datetime(sp500Data.index).date
        sp500Data.index = sp500DataIndex
        resultingData = pd.merge(perfDataRollingYear, sp500Data, left_index = True, right_index = True)
        resultingData = resultingData/resultingData.iloc[0]
        resultingDataWithChange = pd.merge(resultingData, perfData, left_index = True, right_index = True)

        return resultingDataWithChange

