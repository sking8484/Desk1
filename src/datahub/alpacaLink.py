from csv import excel_tab
from os import times
import pandas as pd
from datetime import date
import requests
from datetime import datetime
import time
import os
from alpaca.data import StockHistoricalDataClient, requests
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


class AlpacaLink:
    def __init__(self, dataLink):
        load_dotenv()
        pass

    def retrieve_stock_name(self, alpaca_historical_bars_df):
        index = alpaca_historical_bars_df.index
        return index[0][0]

    def retrieve_stock_history(self, alpaca_historical_bars_df):
        stock_name = self.retrieve_stock_name(alpaca_historical_bars_df)
        return alpaca_historical_bars_df.loc[stock_name]

    def retrieve_closing_price(self, alpaca_historical_bars_df):
        history = self.retrieve_stock_history(alpaca_historical_bars_df)
        return history[['close']]
    
    def switch_column_names(self, dataframe, names):
        return dataframe.rename(columns = names)

    def join_dataframes(self, df_1, df_2):
        return df_1.join(df_2)

    def transform_alpaca_bars_output(self, bars_df):
        symbol = self.retrieve_stock_name(bars_df)
        closing_price = self.retrieve_closing_price(bars_df)
        return self.switch_column_names(closing_price, {'close':symbol})

    def build_stock_prices_frame(self, df_1=None, df_2=None):
        if df_1 is None:
            return self.transform_alpaca_bars_output(df_2)
        else:
            return self.join_dataframes(df_1, self.transform_alpaca_bars_output(df_2))

    def normalize_data(self, stock_df):
        cols = stock_df.columns
        standard_index_df = stock_df.reset_index()
        correct_col_name_df = self.switch_column_names(standard_index_df, {"timestamp":"date"})
        return pd.melt(correct_col_name_df, id_vars = ['date'], value_vars = cols, var_name="symbol")

    def get_historical_data(self, symbols, start_time, end_time):
        stock_client = StockHistoricalDataClient(api_key=os.environ["API_KEY"], secret_key=os.environ["SECRET_KEY"], use_basic_auth=False)
        req = requests.StockBarsRequest(symbol_or_symbols=symbols, start=start_time, end=end_time, timeframe=TimeFrame.Day)
        return stock_client.get_stock_bars(req).df

