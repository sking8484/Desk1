from abc import ABC, abstractmethod 
import pandas as pd 

class DataAPI(ABC):
    
    @abstractmethod
    def __init__(self, credentials: dict):
        pass 
    
    @abstractmethod
    def create_table(self, tableName: str, dataFrame: pd.DataFrame):
        pass

    @abstractmethod
    def append(self, tableName: str, dataFrame: pd.DataFrame):
        pass 

    @abstractmethod
    def return_table(self, tableName: str):
        pass

    @abstractmethod
    def delete_table(self, tableName: str):
        pass


