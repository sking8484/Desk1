import pandas as pd
import mysql.connector as sqlconnection
from privateKeys.privateData import credentials
import numpy as np

class dataLink:
    def __init__(self):

        '''
        dataLink: Cohesive object for moving data to and from mySQL

        Going to pull in credentials from a private key file, and is going to connect to the mysql database running there.
        '''
        credents = credentials()
        self.cnxn = sqlconnection.connect(**credents.credentials)
        self.cursor = self.cnxn.cursor()
    
    def createTable(self, tableName, dataFrame = "", addAutoIncrementCol = True, addData = True):

        '''
        The create table method, takes in a pandas dataframe and table name and creates a new table.

            tableName: String name for the table. If this table already exists, we will get an error
            dataFrame: Pandas dataFrame which will be turned into a sql table.
        '''
        columnString = ""
        valuesString = ""
        dataFrame = dataFrame.replace(np.nan,0,regex=True)

        for col in dataFrame.columns:
            if (col != dataFrame.columns[-1]):
                columnString += col + " TEXT, "
                valuesString += "%s, "
            else:
                columnString += col + " TEXT"
                valuesString += "%s"
        columnString = columnString.replace(".","_")

        if addAutoIncrementCol:
            query = "CREATE TABLE " + tableName + " (ID int NOT NULL AUTO_INCREMENT, " + columnString + ", PRIMARY KEY (ID))"
        else:
            query = "CREATE TABLE " + tableName + "(" + columnString + ")"

        self.cursor.execute(query)

        if addData:
            colsNoDefinitions = columnString.replace(" TEXT","")
            query = "INSERT INTO " + tableName + " (" + colsNoDefinitions + ") VALUES (" + valuesString + ")"
            
            for i in range(len(dataFrame.index)):
                if i%100 == 0:
                    print(i/len(dataFrame.index))
                self.cursor.execute(query,tuple(dataFrame.iloc[i,].values))
        
        self.commit()


    def joinTables(self, joinColumn, originalTableName, joinTableDataFrame):
        "Need to figure out how to join tables without using select ... into, thinking of using create table like and having the like table be result of query"
        self.createTable('interimTable', joinTableDataFrame, False)
        query = "SELECT * INTO " + originalTableName + "_ FROM " + originalTableName + "LEFT JOIN interimTable ON " + originalTableName + "." + joinColumn + " = interimTable." + joinColumn 
        print(query)
        self.cursor.execute(query)
        self.commit()

    def dropColumns(self, tableName, columnList):
        query = "ALTER TABLE " + tableName + " DROP COLUMN "
        for col in columnList:
            if col != columnList[-1]:
                query += col + ", "
            else:
                query += col + ";"
        
        self.cursor.execute(query)
        self.commit()


    def deleteTable(self, tableName):
        query = "DROP TABLE " + tableName
        self.cursor.execute(query)
        self.commit()


    def commit(self):
        self.cnxn.commit()
