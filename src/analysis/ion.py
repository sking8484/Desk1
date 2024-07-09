import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional
import pandas as pd
import numpy as np
from numpy import transpose as t
import sys
import datetime
from datetime import date
import math
import time
from warnings import simplefilter
from abstract_classes_analysis import AnalysisToolKit, Gerber, QuantMethods, MSSA, RegressionOutput
from sklearn.linear_model import LinearRegression
import logging
"""
The ion class is Desk1's main tool for analyzing data. This is where many of Desk1's trading signal's will be created.
"""
def test_setup():
    print("SETUP")

class QuantTools(QuantMethods):
    
    def __init__(self, data: pd.DataFrame):
        self.data = data 
    
    def get_gerber(self, Q: int) -> Gerber:
        return GerberStatistic(data = self.data, Q = Q)

    def get_mssa(self, L:int, lookBack: Optional[int] = 0, informationThreshold: Optional[float] = .95, useIntercept: Optional[bool] = False) -> MSSA:
        return SpectrumAnalysis(data = self.data, L = L, lookBack = lookBack, informationThreshold = informationThreshold, useIntercept = useIntercept)

    
class AnalysisMethods(AnalysisToolKit):
    
    def calculate_num_rows(self, data: np.ndarray) -> int:
        return data.shape[0]

    def diagonalize_matrix(self, data: np.ndarray) -> np.ndarray:
        return np.diag(data)

    def calculate_std(self, data: np.ndarray, axis: Optional[int] = 0) -> np.ndarray:
        return np.std(data, axis = axis)

    def divide_matrices(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return numerator / denominator

    def calculate_svd(self, matrix: np.ndarray) -> dict[str, np.ndarray]:
        matrix = np.float64(matrix)
        d = np.linalg.matrix_rank(matrix)

        U, Sigma, V = np.linalg.svd(matrix)
        V = t(V)

        X_elem = np.array([ Sigma[i] * np.outer(U[:,i], V[:,i]) for i in range(0,d)])

        return {
            "elementary_matrices":X_elem,
            "singular_values":Sigma
        }

    def filter_svd_matrices(self, elementaryMatrices: np.ndarray, singularValues: np.ndarray, informationThreshold: Optional[float] = .95) -> np.ndarray:
        values = ((singularValues ** 2)/sum(singularValues **2 ))

        information = 0
        filteredMatrix = None 
        for i in range(len(elementaryMatrices)):
            if information >= informationThreshold:
                break
            if information == 0:
                filteredMatrix = elementaryMatrices[i]
            else:
                filteredMatrix = filteredMatrix + elementaryMatrices[i]

            information += values[i]

        return filteredMatrix

    def run_regression(self, features: np.ndarray, labels: np.ndarray, transposeFeatures: Optional[bool] = True, intercept: Optional[bool] = False) -> RegressionOutput:
        if transposeFeatures:
            features = t(features)

        reg = LinearRegression(fit_intercept = intercept).fit(features, labels, )
        returnObj = RegressionOutput(coefficients = reg.coef_, model = reg)
        if intercept:
            returnObj.intercept = reg.intercept_

        return returnObj

    def clean_data(self, data: pd.DataFrame, lookBack: Optional[int] = 0, removeNullCols: Optional[bool] = False, removeDateColumn: Optional[bool] = False, use_change = False) -> pd.DataFrame:
        if removeDateColumn:
            if 'date' in data.columns:
                data.drop(columns = ['date'], inplace = True)
            elif 'Date' in data.columns:
                data.drop(columns = ['Date'], inplace=True)

        if use_change:
            data = data.apply(pd.to_numeric).pct_change().dropna()

        if lookBack != 0:
            data = data[-lookBack:]

        if removeNullCols:
            data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)


        if removeDateColumn:
            if 'date' in data.columns:
                data.drop(columns = ['date'], inplace = True)
            elif 'Date' in data.columns:
                data.drop(columns = ['Date'], inplace=True)

        return data
                    

class SpectrumAnalysis(MSSA, AnalysisMethods):
    
    def __init__(self, data: pd.DataFrame, L: int, lookBack: Optional[int] = 0, informationThreshold: Optional[float] = .95, useIntercept: Optional[bool] = False):
        self.data = data 
        self.L = L 
        if lookBack == 0:
            lookBack = self.calculate_num_rows(data.to_numpy())
        self.lookBack = lookBack 
        if self.lookBack % self.L != 0:
            logging.critical("Lookback not divisible by L")
        self.informationThreshold = informationThreshold
        self.useIntercept = useIntercept

    def create_prediction_features(self, data: np.ndarray, columns: list[str], L: int, lookBack: int) -> dict[str, np.ndarray]:
        returnObj = {}
        columnsPerSeries = int(lookBack / L)
        for columnName, column in zip(columns, range(columnsPerSeries - 1, len(data[1]), columnsPerSeries)):
            returnObj[columnName] = np.array([data[-L + 1:, column]])

        return returnObj

    def create_page_matrix(self, data: np.ndarray, L: int, lookBack: int):
        if lookBack % L != 0:
            logging.critical(f"Lookback: {lookBack} not divisible by {L}")
            return None
            
        K = lookBack - L + 1
        data = data[-lookBack:]
        page_matrix = np.column_stack([data[i:i+L] for i in range(0, K, L)])
        return page_matrix

    def concat_matrices(self, baseMatrix: np.ndarray, additionalMatrix: np.ndarray) -> np.ndarray:
        return np.column_stack((baseMatrix, additionalMatrix))

    def create_hsvt_matrix(self, data: pd.DataFrame, L: int, lookBack: int, informationThreshold: Optional[float] = .95) -> np.ndarray:
        
        ### Clean Data

        hsvt_matrix = None
        self.columns = data.columns 
        for i in range(len(self.columns)):
            page_matrix = self.create_page_matrix(np.array(data[self.columns[i]]), L, lookBack)
            svd_proceedure = self.calculate_svd(page_matrix)
            filtered = self.filter_svd_matrices(svd_proceedure['elementary_matrices'], svd_proceedure['singular_values'], informationThreshold = informationThreshold)

            if i == 0:
                hsvt_matrix = filtered

            else:
                hsvt_matrix = self.concat_matrices(hsvt_matrix, filtered)

        return hsvt_matrix

    def create_labels_features(self, hsvtMatrix: np.ndarray) -> dict[str, np.ndarray]:
        num_rows = self.calculate_num_rows(hsvtMatrix)
        labels = hsvtMatrix[num_rows-1,:]
        features = hsvtMatrix[:num_rows-1,:]

        return {
            'labels':labels,
            'features':features
        }

    def learn_linear_model(self, labels: np.ndarray, features: np.ndarray, intercept: Optional[bool] = False):
        return self.run_regression(features = features, labels = labels, intercept = intercept)

    def predict(self, model: LinearRegression, predictors: dict[str, np.ndarray]):
        
        predictions = {}
        curr_date = datetime.datetime.now()
        predictions[curr_date] = {}
        for predictor in predictors:
            features = predictors[predictor]
            prediction = model.predict(features)
            predictions[curr_date][predictor] = prediction[0]

        return predictions

    def format_return_data(self, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
        df_predictions = pd.DataFrame.from_dict(predictions, orient = 'index').reset_index().rename(columns = {'index':'date'})
        formatted_predictions = pd.melt(df_predictions, id_vars = ['date'], var_name = 'symbol')
        return formatted_predictions
        
    def run_mssa(self) -> pd.DataFrame:
        data = self.clean_data(data = self.data, lookBack = self.lookBack, removeDateColumn = True, removeNullCols = True, use_change=True)
        hsvt_matrix = self.create_hsvt_matrix(data, self.L, lookBack = self.lookBack, informationThreshold = self.informationThreshold)
        prediction_features = self.create_prediction_features(hsvt_matrix, data.columns, self.L, self.lookBack)
        labels_features_dict = self.create_labels_features(hsvt_matrix)
        labels = labels_features_dict['labels']
        features = labels_features_dict['features']
        linear_model = self.learn_linear_model(labels = labels, features = features, intercept = self.useIntercept)
        predictions = self.predict(linear_model.model, predictors = prediction_features)
        formatted_predictions = self.format_return_data(predictions)
        return formatted_predictions
        

class GerberStatistic(Gerber, AnalysisMethods):
    
    def __init__(self, data: pd.DataFrame, Q: int):
        self.data = data 
        self.Q = Q 
    
    def calculate_limits(self, data: np.ndarray, Q: int) -> dict[str, np.ndarray]:
        upperLimit = Q*self.calculate_std(data)
        lowerLimit = -1*Q*self.calculate_std(data)

        return {
            'upperLimit':upperLimit,
            'lowerLimit':lowerLimit
        }


    def initialize_upper_lower_matrices(self, data: np.ndarray, upperLimit: np.ndarray, lowerLimit: np.ndarray) -> dict[str, np.ndarray]:
        upperMatrix = data - upperLimit 
        lowerMatrix = data - lowerLimit 

        return {
            'upperMatrix':upperMatrix,
            'lowerMatrix':lowerMatrix
        }

    def calculate_upper_lower_matrices(self, upperMatrix: np.ndarray, lowerMatrix: np.ndarray) -> dict[str, np.ndarray]:

        upperMatrix[upperMatrix >= 0] = 1
        upperMatrix[upperMatrix < 0] = 0

        lowerMatrix[lowerMatrix <= 0] = -1
        lowerMatrix[lowerMatrix > 0] = 0
        lowerMatrix[lowerMatrix == -1] = 1
        
        return {
            'upperMatrix': upperMatrix,
            'lowerMatrix': lowerMatrix
        }

    def calculate_mid_matrix(self, upperMatrix: np.ndarray, lowerMatrix: np.ndarray) -> np.ndarray:
        
        
        midMatrix = upperMatrix + lowerMatrix
        midMatrix = midMatrix + 1
        midMatrix[midMatrix == 2] = 0

        return midMatrix

    def build_gerber_numerator(self, upperMatrix: np.ndarray, lowerMatrix: np.ndarray) -> np.ndarray:

        N_UU = t(upperMatrix) @ upperMatrix
        N_DD = t(lowerMatrix) @ lowerMatrix
        N_UD = t(upperMatrix) @ lowerMatrix
        N_DU = t(lowerMatrix) @ upperMatrix

        return N_UU + N_DD - N_UD - N_DU

    def build_gerber_denominator(self, midMatrix: np.ndarray, T: int) -> np.ndarray:

        N_NN = t(midMatrix) @ midMatrix
        denom_mat = np.copy(N_NN)
        denom_mat[denom_mat > -100000] = T
        denom_mat = denom_mat - N_NN

        return denom_mat

    def divide_gerber_matrices(self, num_mat: np.ndarray, denom_mat: np.ndarray) -> np.ndarray:

        return self.divide_matrices(num_mat, denom_mat)

    def create_gerber_stat(self, diagonalizedMatrix: np.ndarray, gerberStat: np.ndarray) -> np.ndarray:
        
        ones = np.ones(np.shape(diagonalizedMatrix))
        diag = (np.diag(diagonalizedMatrix))
        np.fill_diagonal(ones, diag)
        return np.multiply(np.multiply(ones, gerberStat), ones)

    def get_gerber_statistic(self) -> pd.DataFrame:
        
        array_data = np.float64(self.data.to_numpy())
        limits = self.calculate_limits(array_data, self.Q)
        upper_lower_matrices = self.initialize_upper_lower_matrices(array_data, limits['upperLimit'], limits['lowerLimit'])
        upper_lower_matrices = self.calculate_upper_lower_matrices(upper_lower_matrices['upperMatrix'], upper_lower_matrices['lowerMatrix'])
        upper_matrix, lower_matrix = upper_lower_matrices['upperMatrix'], upper_lower_matrices['lowerMatrix']
        mid_matrix = self.calculate_mid_matrix(upper_matrix, lower_matrix)
        gerber_numerator = self.build_gerber_numerator(upper_matrix, lower_matrix)
        gerber_denominator = self.build_gerber_denominator(mid_matrix, self.calculate_num_rows(array_data))
        gerber_matrix = self.divide_matrices(gerber_numerator, gerber_denominator)
        gerber_stat = self.create_gerber_stat(self.diagonalize_matrix(self.calculate_std(array_data, axis = 0)), gerber_matrix)

        return pd.DataFrame(gerber_stat, index = self.data.columns, columns = self.data.columns)

class ion:
    def __init__(self):

        """
        A class used to represent a toolkit surrounding the Pandas dataframe data.
        ...

        Methods
        -------
        getGerberMatrix: Returns Pandas DataFrame of the gerber statistic
            (gerberMatrix)
            dimensions M x M

        """

    def getOptimalWeights(self,data:pd.DataFrame, delta:Optional[float] = 50, leverageAmt: Optional[float] = 1.0, gerber:Optional[bool] = True, predictions: Optional[pd.DataFrame] = None, usePredictions: Optional[bool] = False) -> pd.DataFrame:
        """Imports here due to inability to solve enviornment errors"""
        from cvxopt import matrix
        from cvxopt.blas import dot
        from cvxopt.solvers import qp, options
        """
        We are using CVXOPT. The exmaple we are following can be found here
            https://cvxopt.org/examples/book/portfolio.html

        Method to find optimal weights with the equation

        Parameters:
            data: Time series of stock prices WITH DATE COLUMN
            delta: The amount of risk we want to take
            leverageAmt: The amount of leverage we want
            gerber: Whether we should use the gerber matrix or not

        """

        yearAgo = datetime.datetime.now() - datetime.timedelta(days=365)
        data = data[data['date']>datetime.datetime.strftime(yearAgo,"%Y-%m-%d")]
        data = data.drop(columns = ['date'])
        data = data.astype(float)
        data.replace(to_replace=0,method='ffill', inplace=True)
        cleaned_data = data.pct_change().dropna()

        if usePredictions:
            print("using predictions")
            cleaned_data = cleaned_data[predictions['symbol']]
            N = len(cleaned_data.columns)
            prediction_values = np.float64(predictions['value'].values)
            returns = matrix(np.reshape(prediction_values,(N,1)))
        else:
            N = len(cleaned_data.columns)
            returns = matrix(np.reshape(cleaned_data.mean().values,(N,1)))



        #comovement = matrix(self.getGerberMatrix(cleaned_data, 0.1).values)
        
        comovement = matrix(cleaned_data.cov().values)


        G1 = matrix(0.0,(N,N))
        G1[::N+1] = -1.0
        G2 = matrix(0.0,(N,N))
        G2[::N + 1] = 1.0
        G = matrix(np.concatenate([G1,G2]))
        #print(G)

        h1 = matrix(0.0,(N,1))
        h2 = matrix(.1*leverageAmt, (N,1))
        h = matrix(np.concatenate([h1,h2]))
        #print(G)
        #print(h)
        A = matrix(1.0,(1,N))
        b = matrix(leverageAmt)


        weights = qp(delta*comovement,-returns, G,h,A,b)['x']
        weights = np.round(weights,decimals=3)

        weights = pd.DataFrame(weights, columns = ['value'])
        weights['date'] = datetime.datetime.now()
        weights['symbol'] = cleaned_data.columns

        weights = weights[['date', 'symbol', 'value']]
        return weights

    def getGerberMatrix(self, data, q):
        gerberStat = GerberStatistic(data, q)
        stat = gerberStat.get_gerber_statistic()
        return stat
