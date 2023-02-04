from ipaddress import collapse_addresses
import pandas as pd
import mysql.connector as sqlconnection
from inspect import trace
import numpy as np
import traceback

"""TODO - Create ABC of dataLink. This will allow for better testing"""

class DataLink:
    def __init__(self, credentials: dict):

        """
        A class used to create a datalink between Python (Pandas specifically) and SQL.
        ...
        """
        credentials['autocommit'] = True
        self.cnxn = sqlconnection.connect(**credentials)
        self.cursor = self.cnxn.cursor()

    def createTable(self, tableName: str, dataFrame: pd.DataFrame, addAutoIncrementCol = True) -> None:

        """
        A method to create a table, using mysql connector and executemany.
        ...

        Attributes
        ----------
        tableName: The name of the table you'd like to create
        dataFrame: The dataFrame you'd like to upload to mySQL
        addAutoIncrementCol: Can be used to forego adding autoincrement when building table.
                            Really only used from other calls. When creating tables, this should be True
        """
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

        query = "CREATE TABLE " + tableName + "(" + columnString + ")"

        self.cursor.execute(query)

    def append(self, tableName: str, dataFrame: pd.DataFrame) -> None:

        try:
            columnString = ""
            valuesString = ""
            dataFrame = dataFrame.replace(np.nan,0,regex=True)

            for col in dataFrame.columns:
                if (col != dataFrame.columns[-1]):
                    columnString += col + ", "
                    valuesString += "%s, "
                else:
                    columnString += col
                    valuesString += "%s"
            columnString = columnString.replace(".","_")

            query = "INSERT INTO " + tableName + " (" + columnString + ") VALUES (" + valuesString + ")"

            self.cursor.executemany(query,dataFrame.values.tolist())

        except Exception as e:

            print(traceback.print_exc())
            print("Could not find table. Creating now")
            self.createTable(tableName, dataFrame)

    def returnTable(self, tableName: str, pivotObj:dict = None) -> pd.DataFrame:

        query = "SELECT * FROM " + tableName

        try:
            self.cursor.execute(query)
            out = self.cursor.fetchall()
        except Exception as e:
            return pd.DataFrame({'date':[]})

        db = pd.DataFrame(out)
        field_names = [i[0] for i in self.cursor.description]
        db.columns = field_names

        if pivotObj != None:
            return db.pivot(index = pivotObj['index'], columns = pivotObj['columns'], values = pivotObj['values']).reset_index()
        else:
            return db

    def dropColumns(self, tableName, columnList):

        """
        A method to drop columns
        ...

        Attributes
        ----------
        tableName: The table to drop columns in
        columnList: A list of columns. If only one item, include in list still, e.g. ["AAPL"]
        """
        if len(columnList) == 0:
            return

        query = "ALTER TABLE " + tableName + " DROP COLUMN "
        for col in columnList:
            if col != columnList[-1]:
                query += col + ", DROP COLUMN "
            else:
                query += col + ";"

        self.cursor.execute(query)


    def deleteTable(self, tableName):

        """
        A method to delete a table
        ...

        Attributes
        ----------
        tableName: The name of the table to delete
        """

        query = "DROP TABLE " + tableName
        self.cursor.execute(query)


    

    def getColumns(self, table:str) -> list:
        return self.getLastRow(table).columns


    def getLastRow(self, table:str) -> pd.DataFrame:

        query = "SELECT * FROM " + table + " WHERE date = (SELECT MAX(date) FROM " + table + ")"
        self.cursor.execute(query)
        out = self.cursor.fetchall()
        temp_df = pd.DataFrame(out)
        field_names = [i[0] for i in self.cursor.description]
        temp_df.columns = field_names
        return temp_df

    def getUniqueSymbols(self, table:str) -> list:
        query = "SELECT symbol FROM " + table + " GROUP BY symbol"
        self.cursor.execute(query)
        out = self.cursor.fetchall()
        columnList = list(pd.DataFrame(out).iloc[:,0])
        return columnList

    def getAggElement(self, table:str, column:str, aggFunc:str, conditional:dict):
        '''
        conditional = {"column":column,
                       "value":value}
        '''

        query = "SELECT " + aggFunc + "(" + column + ")" + " FROM " + table
        if len(conditional) > 0:
            query += " WHERE " + conditional['column'] + " = " + "'" + conditional['value'] + "'"
        self.cursor.execute(query)
        out = self.cursor.fetchall()[0][0]
        return out

    def closeConnection(self):
        self.cnxn.close()
