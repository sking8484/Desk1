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

    @abstractmethod
    def calculate_svd(self, matrix: np.ndarray) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def filter_svd_matrices(self, elementaryMatrices: np.ndarray, singularValues: np.ndarray, limit: int) -> np.ndarray:
        pass

    @abstractmethod
    def run_regression(self, features: np.ndarray, labels: np.ndarray, transposeFeatures: Optional[bool] = True, intercept: Optional[bool] = False):
        pass

class MSSA(ABC):
    
    @abstractmethod
    def __init__(self, data: pd.DataFrame, L: int, lookBack: int, informationThreshold: Optional[float] = .95):
        pass 

    @abstractmethod
    def create_page_matrix(self, data: np.ndarray, L: int, lookBack: int) -> np.ndarray:
        pass 

    @abstractmethod
    def concat_matrices(self, baseMatrix: np.ndarray, additionalMatrix: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def create_hsvt_matrix(self, TBD: np.ndarray):
        pass

    @abstractmethod
    def create_labels_features(self, hsvtMatrix: np.ndarray):
        pass

    @abstractmethod
    def learn_linear_model(self, labels: np.ndarray, features: np.ndarray):
        pass

    @abstractmethod
    def predict(self, learned_params: np.ndarray, predictors: np.ndarray):
        pass

    
class Gerber(ABC):
    
    @abstractmethod
    def __init__(self, data: pd.DataFrame, Q: int):
        pass

    @abstractmethod
    def calculate_limits(self, data: np.ndarray, Q: int) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def initialize_upper_lower_matrices(self, data: np.ndarray, upperLimit: np.ndarray, lowerLimit: np.ndarray) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def calculate_upper_lower_matrices(self, upperMatrix: np.ndarray, lowerMatrix: np.ndarray) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def calculate_mid_matrix(self, upperMatrix: np.ndarray, lowerMatrix: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def build_gerber_numerator(self, upperMatrix: np.ndarray, lowerMatrix: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def build_gerber_denominator(self, middleMatrix: np.ndarray, T: int) -> np.ndarray:
        pass
        
    @abstractmethod
    def divide_gerber_matrices(self, num_mat: np.ndarray, denom_mat: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def create_gerber_stat(self, diagonalizedMatrix: np.ndarray, gerberStat: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_gerber_statistic(self) -> pd.DataFrame:
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
