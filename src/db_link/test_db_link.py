from db_link.db_link import DataLink 
import unittest 
from unittest.mock import patch
import pandas as pd

class TestDataLink(unittest.TestCase):
    
    TABLE_NAME = "DARST"

    def get_fake_df(self):
        d = {'col1': [1, 2], 'col.2': [3, 4]}
        df = pd.DataFrame(data=d)
        return df

    @patch('db_link.db_link.sqlconnection.connect')
    def test_init(self, mockConnection):
        link = DataLink()
        mockConnection.assert_called_once()

    @patch('db_link.db_link.sqlconnection.connect')
    def test_create_table(self, mockCursor):
        df = self.get_fake_df()

        connection = mockCursor.return_value
        cursor = connection.cursor.return_value

        link = DataLink()
        link.create_table(self.TABLE_NAME, df)

        sql = f"CREATE TABLE IF NOT EXISTS {self.TABLE_NAME}(col1 TEXT, col_2 TEXT)"

        cursor.execute.assert_called_once()
        cursor.execute.assert_called_with(sql)

    @patch('db_link.db_link.sqlconnection.connect')
    def test_append_table_not_exists(self, mockConnection):
        df = self.get_fake_df()

        conn = mockConnection.return_value
        cursor = conn.cursor.return_value

        link = DataLink()
        link.append(self.TABLE_NAME, df)

        sql = f"INSERT INTO {self.TABLE_NAME} (col1, col_2) VALUES (%s, %s)"
        data = df.values.tolist()

        create_table_sql = f"CREATE TABLE IF NOT EXISTS {self.TABLE_NAME}(col1 TEXT, col_2 TEXT)"

        cursor.execute.assert_called_once()
        cursor.execute.assert_called_with(create_table_sql)
        cursor.executemany.assert_called_once()
        cursor.executemany.assert_called_with(sql, data)

    @patch('db_link.db_link.DataLink.create_table')
    @patch('db_link.db_link.sqlconnection.connect')
    def test_append_table_exists(self, mockConnection, mockCreateTable):
        df = self.get_fake_df()

        conn = mockConnection.return_value
        cursor = conn.cursor.return_value
        
        mockCreateTable.side_effect = KeyError('Foo')

        link = DataLink()
        link.append(self.TABLE_NAME, df)

        sql = f"INSERT INTO {self.TABLE_NAME} (col1, col_2) VALUES (%s, %s)"
        data = df.values.tolist()

        cursor.execute.assert_not_called()
        cursor.executemany.assert_called_once()
        cursor.executemany.assert_called_with(sql, data)

    @patch('db_link.db_link.sqlconnection.connect')
    def test_return_table(self, mockConnection):
        conn = mockConnection.return_value
        cursor = conn.cursor.return_value
        cursor.fetchall.return_value = self.get_fake_df().to_dict()
        cursor.description = [["col1"], ["col.2"]]

        link = DataLink()
        table = link.return_table(self.TABLE_NAME)

        sql = f"SELECT * FROM {self.TABLE_NAME}"

        cursor.execute.assert_called_once()
        cursor.execute.assert_called_with(sql)

        cursor.fetchall.assert_called_once()
        
        self.assertEqual(table.to_dict(), self.get_fake_df().to_dict())

    @patch('db_link.db_link.sqlconnection.connect')
    def test_drop_columns(self, mockConnection):
        conn = mockConnection.return_value
        cursor = conn.cursor.return_value

        link = DataLink()
        link.drop_columns(self.TABLE_NAME, [])

        cursor.execute.assert_not_called()

        colList = ["COL1"]
        sql = f"ALTER TABLE {self.TABLE_NAME} DROP COLUMN COL1;"
        link.drop_columns(self.TABLE_NAME, colList)

        cursor.execute.assert_called_once()
        cursor.execute.assert_called_with(sql)


if __name__ == "__main__":
    unittest.main()
