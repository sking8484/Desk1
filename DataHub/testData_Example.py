# How to get stock data from IEX to Google
# Pulling private data from gitignore file
from pickle import FALSE
from posixpath import split
import sys 
from privateKeys.privateData import credentials
import pandas as pd
import requests
import mysql.connector as sqlconnection
from dataLink import dataLink
import numpy as np
from iexLink import iexLink

"""
cursor.execute("SELECT * FROM testData9")
out = cursor.fetchall()
db = pd.DataFrame(out)

## Get the column names from the datapull
field_names = [i[0] for i in cursor.description]
db.columns = field_names

print(db['dateVal'].values)

"""

credents = credentials()

testData = pd.read_csv(credents.stockDataPath, index_col = 0)




connection = dataLink(credents.credentials)
connection.dropColumns('test1',['AAPL','MSFT'])

#link = iexLink(credents.iexToken)
#link.getStockData(['AAPL', 'TSLA'],"20220331")






