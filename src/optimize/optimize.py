import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abstract_classes_optimize import Optimizer
import pandas as pd
from analysis.abstract_classes_analysis import QuantMethods
from db_link.abstract_classes_db_link import DataAPI
import numpy as np
from typing import Optional, List



class PortfolioOptimizer(Optimizer):
     
    def __init__(self, data: pd.DataFrame, analysisPKG: QuantMethods, dataLink: DataAPI, delta:Optional[float] = 50, leverageAmt: Optional[float] = 1, timeDelta: Optional[int] = 252, predictions: Optional[pd.DataFrame] = None, usePredictions: Optional[bool] = False, useGerber: Optional[bool] = False):
        pass
