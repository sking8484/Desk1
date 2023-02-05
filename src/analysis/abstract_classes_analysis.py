from abc import ABC, abstractmethod 
import pandas as pd 
import numpy as np 
from typing import List, Optional
from db_link.abstract_classes_db_link import DataAPI

class AnalysisToolKit(ABC):
    
    @abstractmethod
    def calculate_num_rows(self, data: np.ndarray) -> int:
        pass

    @abstractmethod
    def diagonalize_matrix(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_std(self, data: np.ndarray, axis: Optional[int] = 0) -> np.ndarray:
        pass

    @abstractmethod
    def divide_matrices(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        pass

class Gerber(ABC):
    
    @abstractmethod
    def __init__(self, data: pd.DataFrame, Q: int):
        pass

    @abstractmethod
    def calculate_limits(self, data: pd.DataFrame, Q: int) -> dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def initialize_upper_lower_matrices(self, data: pd.DataFrame) -> dict[str, np.matrix]:
        pass

    @abstractmethod
    def calculate_upper_lower_matrices(self, upperMatrix: np.matrix, lowerMatrix: np.matrix) -> dict[str, np.matrix]:
        pass

    @abstractmethod
    def calculate_mid_matrix(self, upperMatrix: np.matrix, lowerMatrix: np.matrix) -> np.matrix:
        pass

    @abstractmethod
    def build_gerber_numberator(self, upperMatrix: np.matrix, midMatrix: np.matrix, lowerMatrix: np.matrix) -> dict[str, np.matrix]:
        pass

    @abstractmethod
    def build_gerber_denominator(self, data: np.matrix, T: int) -> np.matrix:
        pass
        
    @abstractmethod
    def create_gerber_matrix(self, diagonalizedMatrix: np.matrix, divisionRemainder: np.ndarray) -> np.matrix:
        pass

    @abstractmethod
    def get_gerber_statistic(self):
        pass


class QuantMethods(ABC):
    @abstractmethod 
    def __init__(self, data: pd.DataFrame, dataLink: DataAPI):
        pass

    @abstractmethod
    def get_gerber(self, Q: int) -> Gerber:
        pass

    #@abstractmethod
    #def get_mahalanobis(self) -> Mahalanobis:
    #    pass
