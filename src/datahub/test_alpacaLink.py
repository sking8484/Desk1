import unittest 
import numpy as np
from unittest.mock import patch
import pandas as pd
from datahub.alpacaLink import AlpacaLink
from db_link.db_link import DataLink
import alpaca.data as apdata

class TestAlpacaLink(unittest.TestCase):
    
    def return_test_data(self, i,j):
        return [[n+1 for m in range(j)] for n in range(i)]
    def get_fake_stock_bars(self, name):
        arrays = [
            [name for i in range(8)],
            ["01/01/2020", "01/02/2020", "01/03/2020", "01/04/2020", "01/05/2020", "01/06/2020", "01/07/2020", "01/08/2020"],
        ]

        tuples = list(zip(*arrays))

        index = pd.MultiIndex.from_tuples(tuples, names=["symbol", "timestamp"])

        df = pd.DataFrame(self.return_test_data(8,4), columns = ["high", "low", "open", "close"], index = index)
        return df

    #def test_stock_table_conversion(self):
    #    expected = self.get_fake_stock_bars()
    #
    #
    #
    def concatted_stock_table(self):
        array = ["01/01/2020", "01/02/2020", "01/03/2020", "01/04/2020", "01/05/2020", "01/06/2020", "01/07/2020", "01/08/2020"]
        df = pd.DataFrame(self.return_test_data(8,2), columns = ["close1", "close2"], index = array)

        return df
    
    def transformed_stock_table(self):
        index = pd.Index(["01/01/2020", "01/02/2020", "01/03/2020", "01/04/2020", "01/05/2020", "01/06/2020", "01/07/2020", "01/08/2020"],name="timestamp")

        test_data = self.return_test_data(8,2)
        df = pd.DataFrame(test_data, columns = ["spy", "aapl"], index = index)
        return df

    def build_alpaca_link(self):
        return AlpacaLink(DataLink)

    def test_retrieve_name_from_index(self):
        name = 'spy'
        input = self.get_fake_stock_bars(name)
        expected = name

        alpacaLink = self.build_alpaca_link()
        output = alpacaLink.retrieve_stock_name(input)
        self.assertEqual(output, expected)

    def test_retrieve_stock_from_table(self):
        input = self.get_fake_stock_bars('spy')
        expected = input.loc['spy']
        
        alpacaLink = self.build_alpaca_link()
        output = alpacaLink.retrieve_stock_history(input)
        pd.testing.assert_frame_equal(output, expected)

    def test_retrieve_closing_price_from_table(self):
        input = self.get_fake_stock_bars('spy')
        expected = input.loc['spy'][['close']]

        alpacaLink = self.build_alpaca_link()
        output = alpacaLink.retrieve_closing_price(input)
        pd.testing.assert_frame_equal(output, expected)

    def test_switch_column_names(self):
        input = self.get_fake_stock_bars('spy').loc['spy'][['close']]
        expected = input.rename(columns = {"close":"spy"})

        alpacaLink = self.build_alpaca_link()
        output = alpacaLink.switch_column_names(input, {'close':'spy'})
        pd.testing.assert_frame_equal(output, expected)

    def test_concat_dataframes(self):
        expected = self.concatted_stock_table()
        input1 = expected[['close1']]
        input2 = expected[['close2']]

        alpacaLink = self.build_alpaca_link()
        output = alpacaLink.join_dataframes(input1, input2)
        pd.testing.assert_frame_equal(output, expected)

    def test_transform_alpaca_output(self):
        expected = self.transformed_stock_table()[['spy']]

        input = self.get_fake_stock_bars('spy')
        alpacaLink = self.build_alpaca_link()
        output = alpacaLink.transform_alpaca_bars_output(input)
        pd.testing.assert_frame_equal(output, expected)

    def test_building_prices_frame(self):
        expected = self.transformed_stock_table()

        input_1 = self.get_fake_stock_bars('spy')
        input_2 = self.get_fake_stock_bars('aapl')

        alpacaLink = self.build_alpaca_link()
        output = alpacaLink.build_stock_prices_frame(alpacaLink.build_stock_prices_frame(df_2 = input_1), input_2)

        pd.testing.assert_frame_equal(output, expected)

    def test_normalized_data(self):
        df = self.transformed_stock_table()
        nonmelt_expected = df.reset_index().rename(columns = {"timestamp":"date"})
        expected = pd.melt(nonmelt_expected, id_vars = ['date'], value_vars = ['spy','aapl'], var_name = "symbol")

        alpacaLink = self.build_alpaca_link()
        output = alpacaLink.normalize_data(df)

        pd.testing.assert_frame_equal(output, expected)

    '''def test_integration_get_data(self):
        symbols = ['aapl', 'tsla']
        alpacaLink = self.build_alpaca_link()
        start = pd.to_datetime("12/01/2020")
        end = pd.to_datetime("12/01/2021")
        resp = alpacaLink.get_historical_data(symbols, start, end)
        print(resp)
    '''

if __name__=="__main__":
    unittest.main()

