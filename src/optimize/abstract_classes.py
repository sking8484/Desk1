from abc import ABC, abstractmethod 
import pandas as pd 
import numpy as np
from typing import Optional
from datetime import datetime


class Optimize(ABC):
    
    @abstractmethod
    def __init__(self, data: pd.DataFrame, analysisPKG: QuantMethods, delta:Optional[float] = 50, leverageAmt: Optional[float] = 1):
        pass

    @abstractmethod
    def cleanData(self) -> pd.DataFrame:
        pass 

    @abstractmethod
    def calculate_time_delta(self, num_days: int) -> datetime.datetime:
        pass

    @abstractmethod
    def create_returns(self, predictions: Optional[pd.DataFrame] = None, usePredictions: Optional[bool] = False):
        pass

    @abstractmethod 
    def calculate_comovement(self, useGerber: Optional[bool] = True):
        pass

    @abstractmethod
    def setup_cvxopt(self) -> List[pd.DataFrame]:
        pass

    @abstractmethod
    def calculate_weights(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_optimal_weights(self) -> pd.DataFrame:
        pass

