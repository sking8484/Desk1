import pandas as pd
import mysql.connector as sqlconnection
from privateKeys.privateData import credentials
import numpy as np

class dataLink:
    def __init__(self):

        """
        A class used to create a datalink between Python (Pandas specifically) and SQL.
        ...
        """

        credents = credentials()
        self.cnxn = sqlconnection.connect(**credents.credentials)
        self.cursor = self.cnxn.cursor()
    
    def createTable(self, tableName, dataFrame, addAutoIncrementCol = True):

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

        if addAutoIncrementCol:
            query = "CREATE TABLE " + tableName + " (ID int NOT NULL AUTO_INCREMENT, " + columnString + ", PRIMARY KEY (ID))"
        else:
            query = "CREATE TABLE " + tableName + "(" + columnString + ")"

        self.cursor.execute(query)

        
        colsNoDefinitions = columnString.replace(" TEXT","")
        query = "INSERT INTO " + tableName + " (" + colsNoDefinitions + ") VALUES (" + valuesString + ")"
        
        self.cursor.executemany(query,dataFrame.values.tolist())
        
        self.commit()


    def joinTables(self, joinColumn, originalTableName, joinTableDataFrame):

        """
        A method to join two tables
            Creates a new table with same table name, but now with joined columns

            1. Creates an interim table using the createTable method without autoIncs. 
            2. Creates a new pandas dataframe from the two joins
            3. Creates a new table called originalTableName_ from the dataframe
            4. Deletes originalTableName table
            5. Sets old table to originalTableName
            6. Deletes originalTableName_
        ...

        Attributes
        ----------
        joinColumn: The primary key to join on.
                    Must be in both tables
        originalTableName: The table to be ammended
        joinTableDataFrame: The new data that should be joined
        """

        #1.) Create interim table 
        self.createTable('interimTable', joinTableDataFrame, False)

        #2.) Create new pandas dataframe from joins
        query = "SELECT * FROM " + originalTableName + " LEFT JOIN interimTable ON " + originalTableName + "." + joinColumn + " = interimTable." + joinColumn
        self.cursor.execute(query)

        out = self.cursor.fetchall()
        temp_df = pd.DataFrame(out) 
        field_names = [i[0] for i in self.cursor.description]
        temp_df.columns = field_names

        temp_df = temp_df.loc[:,~temp_df.columns.duplicated()]
        #3.) Creates new table with joined dataframe

        self.createTable(originalTableName + "_", temp_df, False)

        #4.) Deletes old table
        self.deleteTable(originalTableName)

        #5.) Create new table with original table name and set add all values from originaltablename_ table
        query = "CREATE TABLE " + originalTableName + " LIKE " + originalTableName + "_"
        self.cursor.execute(query)
        query = "INSERT INTO " + originalTableName + " SELECT * FROM " + originalTableName + "_"
        self.cursor.execute(query)

        #6.) Delete _ table
        self.deleteTable(originalTableName + "_")
        self.deleteTable("interimTable")

        self.commit()

    def dropColumns(self, tableName, columnList):

        """
        A method to drop columns
        ...

        Attributes
        ----------
        tableName: The table to drop columns in
        columnList: A list of columns. If only one item, include in list still, e.g. ["AAPL"]
        """

        query = "ALTER TABLE " + tableName + " DROP COLUMN "
        for col in columnList:
            if col != columnList[-1]:
                query += col + ", "
            else:
                query += col + ";"
        
        self.cursor.execute(query)
        self.commit()


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
        self.commit()


    def commit(self):

        """
        A method to commit the cursor
        ...

        Attributes
        ----------
        """

        self.cnxn.commit()
