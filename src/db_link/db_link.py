import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipaddress import collapse_addresses
import pandas as pd
import mysql.connector as sqlconnection
from inspect import trace
import numpy as np
import traceback
import abstract_classes

"""TODO - Create ABC of dataLink. This will allow for better testing"""

class DataLink(abstract_classes.DataAPI):

    def __init__(self, credentials: dict):

        """
        A class used to create a datalink between Python (Pandas specifically) and SQL.
        ...
        """
        credentials['autocommit'] = True
        self.cnxn = sqlconnection.connect(**credentials)
        self.cursor = self.cnxn.cursor()

    def create_table(self, tableName: str, dataFrame: pd.DataFrame, addAutoIncrementCol = True) -> None:

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

        query = "CREATE TABLE IF NOT EXISTS " + tableName + "(" + columnString + ")"

        self.cursor.execute(query)

    def append(self, tableName: str, dataFrame: pd.DataFrame) -> None:

        try:
            self.create_table(tableName, dataFrame)

        except Exception as e:
            print("Table Exists")

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

    def return_table(self, tableName: str, pivotObj:dict = None) -> pd.DataFrame:

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

    def drop_columns(self, tableName, columnList):
        
        if len(columnList) == 0:
            return

        query = "ALTER TABLE " + tableName + " DROP COLUMN "
        for col in columnList:
            if col != columnList[-1]:
                query += col + ", DROP COLUMN "
            else:
                query += col + ";"

        self.cursor.execute(query)

    def delete_table(self, tableName):
        
        query = "DROP TABLE " + tableName
        self.cursor.execute(query)

    def get_columns(self, table:str) -> list:
        return self.getLastRow(table).columns

    def get_last_row(self, table:str) -> pd.DataFrame:

        query = "SELECT * FROM " + table + " WHERE date = (SELECT MAX(date) FROM " + table + ")"
        self.cursor.execute(query)
        out = self.cursor.fetchall()
        temp_df = pd.DataFrame(out)
        field_names = [i[0] for i in self.cursor.description]
        temp_df.columns = field_names
        return temp_df

    def get_unique_symbols(self, table:str) -> list:
        query = "SELECT symbol FROM " + table + " GROUP BY symbol"
        self.cursor.execute(query)
        out = self.cursor.fetchall()
        columnList = list(pd.DataFrame(out).iloc[:,0])
        return columnList

    def get_agg_element(self, table:str, column:str, aggFunc:str, conditional:dict):
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
    
def test_setup():
    print("SETUP")
