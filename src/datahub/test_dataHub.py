import unittest 
import numpy as np
from unittest.mock import patch
import pandas as pd
import alpaca.data as apdata
from datahub.dataHub import dataHub

class TestAlpacaLink(unittest.TestCase):
    @patch('db_link.db_link.DataLink')
    def build_dataHub(self, dbLinkPatch):
        from datahub.alpacaLink import AlpacaLink
        from db_link.db_link import DataLink

        return dataHub(dbLinkPatch)
    
    '''
    def test_get_buy_universe(self):
        hub = self.build_dataHub()
        print(hub.getBuyUniverse("TEST_STOCK_TABLE"))
    '''


    @patch('db_link.db_link.DataLink')
    @patch.object(dataHub, 'getBuyUniverse')
    def test_update_time_series_data(self, MockMethod, MockDBLink):
        MockMethod.return_value = ['aapl', 'tsla']
        hub = self.build_dataHub()
        hub.maintainUniverse()



    
    
if __name__=="__main__":
    unittest.main()

