import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abstract_classes_optimize import Optimizer
import pandas as pd
from analysis.abstract_classes_analysis import QuantMethods
from analysis.ion import QuantTools
from db_link.abstract_classes_db_link import DataAPI
import numpy as np
from typing import Optional, List
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
import datetime




class PortfolioOptimizer(Optimizer):
     
    def __init__(self, data: pd.DataFrame, analysisPKG: QuantMethods, dataLink: DataAPI, delta:Optional[float] = 50, leverageAmt: Optional[float] = 1, timeDelta: Optional[int] = 252, predictions: Optional[pd.DataFrame] = None, usePredictions: Optional[bool] = False, useGerber: Optional[bool] = False):
        
        self.data = data
        self.analysisPKG = analysisPKG 
        self.dataLink = dataLink
        self.delta = delta 
        self.leverageAmt = leverageAmt
        self.timeDelta = timeDelta
        self.predictions = predictions
        self.usePredictions = usePredictions 
        self.useGerber = useGerber

    def find_date_column(self, data: pd.DataFrame) -> str:
        
        if 'date' in data.columns:
            return 'date'
        elif 'Date' in data.columns:
            return 'Date'
        else:
            raise TypeError("No Date Column detected")
            
    def filter_data_by_date(self, data: pd.DataFrame, daysBack: datetime) -> pd.DataFrame:

        dateColumn = self.find_date_column(data)
        backDate = self.calculate_time_delta(daysBack)
        data = data[data[dateColumn]>datetime.datetime.strftime(backDate,"%Y-%m-%d")]
        data = data.drop(columns = [dateColumn])
        return data
        
    def calculate_time_delta(self, timeDelta: int) -> datetime:
        backDate = datetime.datetime.now() - datetime.timedelta(days=timeDelta)
        return backDate

    def cleanData(self, data: pd.DataFrame, daysBack: int) -> pd.DataFrame:
        
        data = self.filter_data_by_date(data, daysBack)
        data = data.astype(float)
        data.replace(to_replace=0,method='ffill', inplace=True)
        cleaned_data = data.pct_change().dropna() 
        return cleaned_data

    def align_predictions_and_movements(self, data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        return data[predictions['symbol']]

    def create_returns(self, usePredictions: Optional[bool] = True):

        if usePredictions:
            cleaned_data = cleaned_data[predictions['symbol']]
            N = len(cleaned_data.columns)
            returns = matrix(np.reshape(predictions['value'].values,(N,1)))
        else:
            N = len(cleaned_data.columns)
            returns = matrix(np.reshape(cleaned_data.mean().values,(N,1)))

        return returns

    def calculate_comovement(self):
        pass

    def setup_cvxopt(self) -> List[pd.DataFrame]:
        pass

    def calculate_weights(self) -> pd.DataFrame:
        pass

    def get_optimal_weights(self) -> pd.DataFrame:
        pass
