from privateKeys.privateData import credentials
from dataLink import dataLink
import pandas as pd 
from datetime import date
import requests

"""
Grab last item in stock table in order to get the date
get list of business days using pandas
at each business day, append to stock table, and then append stock table to larger stock table.
then upload this to sql
"""

class iexLink:
    def __init__(self, token:str):
        credents = credentials()
        self.token = token
        self.dataLink = dataLink(credents.credentials)

    def getStockData(self, tickers: list, startDate: str, sinceSpecificDate = True) -> pd.DataFrame:

        if sinceSpecificDate:
            firstJoin = True
            BaseUrl = "https://cloud.iexapis.com/"
            version = "stable/"
            token = "?token=" + self.token
            busDays = self.getBusinessDays(startDate)
            print(busDays)
            for stock in tickers:
                firstDate = True
                for day in busDays:
                    dayStr = day.strftime("%Y%m%d")
                    myParams = "stock/" + stock + "/chart/date/" + dayStr
                    base_url = BaseUrl + version + myParams + token + "&chartCloseOnly=true&chartByDay=true"
                    data = requests.get(base_url)
                    
                    stockData = pd.DataFrame.from_dict(data.json(), orient="columns")[['date','close']]
                    
                    stockData.columns = ['date', stock]

                    if firstDate == True:
                        singleStock = stockData 
                        firstDate = False
                    else:
                        singleStock = pd.concat([singleStock,stockData], axis = 0)
                    
                    
                if firstJoin == True:
                    historicalData = singleStock
                    firstJoin = False
                else:
                    historicalData = historicalData.merge(singleStock, on = 'date', how = 'left')
            
            
    def getBusinessDays(self, startDate):
        return pd.bdate_range(startDate, date.today())
 


