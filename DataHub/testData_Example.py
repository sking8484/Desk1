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
testData = testData.replace(np.nan,0,regex=True)


connection = dataLink()
connection.createTable("testStockTable6",testData)







