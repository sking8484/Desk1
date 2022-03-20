# How to get stock data from IEX to Google
# Pulling private data from gitignore file
from posixpath import split
import sys 
sys.path.insert(0,"/Users/darstking/Desktop/Data/CMF/Finance/Trading/Desk1/privateKeys")
from privateData import credentials
import pandas as pd
import requests
import mysql.connector as sqlconnection

credents = credentials()
## How to get stock data
sandboxBaseUrl = "https://sandbox.iexapis.com/"
version = "stable/"
token = "?token=%s&chartCloseOnly=true" % credents.iexToken
myParams = "stock/TQQQ/chart/6m"
base_url = sandboxBaseUrl + version + myParams + token

data = requests.get(base_url).json()
df = pd.DataFrame.from_dict(data)[['date','close']]

## How to connect to google cloud mysql database



"""
To Create a database, remove the database name in configs. Then connect without that table, and then add the new table into the connection configs.
"""

##creating new database
#cursor = cnx.cursor()
#query1 = "CREATE DATABASE testdb"
#cursor.execute(query1)
#cnx.close()


## create a connection
cnxn = sqlconnection.connect(**credents.testCredentials)
cursor = cnxn.cursor()

#create a table
query2 = "CREATE TABLE testData2 (\
                        date VARCHAR(255), \
                        close FLOAT(10))"

#cursor.execute(query2)
#cnxn.commit()

## Add data to a table 
query3 = ("INSERT INTO testData2 (date, close) VALUES (%s, %s)")
for i in range(len(df.index)):
    cursor.execute(query3 % tuple(df.iloc[i,].values))
cnxn.commit()

cursor.execute("SELECT * FROM testData2")
out = cursor.fetchall()
db = pd.DataFrame(out)

## Get the column names from the datapull
field_names = [i[0] for i in cursor.description]
db.columns = field_names

print(db)











