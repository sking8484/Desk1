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
from alpaca.data.enums import Adjustment
from dotenv import load_dotenv
from pandas.tseries.offsets import BDay


class AlpacaLink:
    def __init__(self, dataLink):
        load_dotenv()
        self.dataLink = dataLink()

    def retrieve_stock_name(self, alpaca_historical_bars_df):
        index = alpaca_historical_bars_df.index
        return index[0][0]

    def retrieve_stock_history(self, alpaca_historical_bars_df):
        stock_name = self.retrieve_stock_name(alpaca_historical_bars_df)
        return alpaca_historical_bars_df.loc[stock_name]

    def retrieve_pricing(self, alpaca_historical_bars_df):
        history = self.retrieve_stock_history(alpaca_historical_bars_df)
        return history[['close', 'open']]
    
    def switch_column_names(self, dataframe, names):
        return dataframe.rename(columns = names)

    def join_dataframes(self, df_1, df_2):
        return pd.concat([df_1, df_2])

    def transform_alpaca_bars_output(self, bars_df):
        symbol = self.retrieve_stock_name(bars_df)
        pricing = self.retrieve_pricing(bars_df)
        pricing["symbol"] = symbol
        return self.normalize_data(pricing)

    def build_stock_prices_frame(self, df_1=None, df_2=None):
        if df_1 is None:
            return self.transform_alpaca_bars_output(df_2)
        else:
            return self.join_dataframes(df_1, self.transform_alpaca_bars_output(df_2))

    def normalize_data(self, stock_df):
        cols = stock_df.columns
        standard_index_df = stock_df.reset_index()
        correct_col_name_df = self.switch_column_names(standard_index_df, {"timestamp":"date"})
        melted_data = pd.melt(correct_col_name_df, id_vars = ['date', 'symbol'], value_vars = cols, var_name="bar_type")
        return melted_data.dropna()

    def get_from_date(self, identifier, table):
        try:
            date = self.dataLink.get_agg_element(table, 'date', 'MAX', {'column':'symbol', 'value':identifier})
        except Exception as e:
            print(e)
            date = '2020-01-01'
        if date == None:
            date =  '2020-01-01'
        fromDate = pd.to_datetime(date)
        return fromDate

    def get_timeseries_data(self, symbols, table):
        first = True
        updated = False
        for symbol in symbols:
            start_date = self.get_from_date(symbol, table)
            end_date = date.today()
            if start_date.date() < pd.to_datetime(end_date - BDay(1)).date():
                updated = True
                stock_data = self.get_historical_data(symbol, start_date + BDay(1), end_date)
                if first:
                    df = self.build_stock_prices_frame(None, stock_data)
                    first = False
                else:
                    df = self.build_stock_prices_frame(df, stock_data)

                print(df)
        if updated:
            return df
        return pd.DataFrame({})


    def get_historical_data(self, symbols, start_time, end_time):
        stock_client = StockHistoricalDataClient(api_key=os.environ["ALPACAPUBKEY"], secret_key=os.environ["ALPACAPRIVKEY"], use_basic_auth=False)
        req = requests.StockBarsRequest(symbol_or_symbols=symbols, start=start_time, end=end_time, timeframe=TimeFrame.Day, adjustment = Adjustment.ALL)
        return stock_client.get_stock_bars(req).df

