from abc import ABC, abstractmethod 
import pandas as pd 
import numpy as np
from typing import Optional, List
from datetime import datetime
from db_link.abstract_classes_db_link import DataAPI
from analysis.abstract_classes_analysis import QuantMethods

class Optimizer(ABC):
    
    @abstractmethod
    def __init__(self, data: pd.DataFrame, analysisPKG: QuantMethods, dataLink: DataAPI, delta:Optional[float] = 50, leverageAmt: Optional[float] = 1, timeDelta: Optional[int] = 252, predictions: Optional[pd.DataFrame] = None, usePredictions: Optional[bool] = False, useGerber: Optional[bool] = False):
        pass

    @abstractmethod
    def find_date_column(self, data: pd.DataFrame) -> str:
        pass

    @abstractmethod
    def cleanData(self) -> pd.DataFrame:
        pass 

    @abstractmethod
    def calculate_time_delta(self) -> datetime:
        pass

    @abstractmethod
    def create_returns(self):
        pass

    @abstractmethod 
    def calculate_comovement(self):
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

