import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    def clean_data(self, data: pd.DataFrame, lookBack: Optional[int] = 0, removeNullCols: Optional[bool] = False, removeDateColumn: Optional[bool] = False) -> pd.DataFrame:
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
        curr_date = datetime.datetime.today().strftime("%Y-%m-%d")
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
        data = self.clean_data(data = self.data, lookBack = self.lookBack, removeDateColumn = True, removeNullCols = True)
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
        
        array_data = self.data.to_numpy()
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
            cleaned_data = cleaned_data[predictions['symbol']]
            N = len(cleaned_data.columns)
            returns = matrix(np.reshape(predictions['value'].values,(N,1)))
        else:
            N = len(cleaned_data.columns)
            returns = matrix(np.reshape(cleaned_data.mean().values,(N,1)))



        comovement = matrix(self.getGerberMatrix(cleaned_data).values)


        G1 = matrix(0.0,(N,N))
        G1[::N+1] = -1.0
        G2 = matrix(0.0,(N,N))
        G2[::N + 1] = 1.0
        G = matrix(np.concatenate([G1,G2]))
        print(G)

        h1 = matrix(0.0,(N,1))
        h2 = matrix(.10, (N,1))
        h = matrix(np.concatenate([h1,h2]))
        print(G)
        print(h)
        A = matrix(1.0,(1,N))
        b = matrix(leverageAmt)

        weights = qp(delta*comovement,-returns, G,h,A,b)['x']
        weights = np.floor(weights*1000)/1000

        weights = pd.DataFrame(weights, columns = ['value'])
        weights['date'] = datetime.datetime.today().strftime("%Y-%m-%d")
        weights['symbol'] = cleaned_data.columns

        weights = weights[['date', 'symbol', 'value']]
        return weights



class orion:
    def __init__(self):
        global tf
        if 'tensorflow' not in sys.modules:
            import tensorflow as tf
        self.tf = tf
        self.TimeRules = TimeRules()
        self.universeCols = []
        self.factorCols = []
        self.windowLength = 252
        self.trained = False


    def updateData(self):
        self.prices = pd.read_csv(self.credents.networkPricesLocation)
        self.factors = pd.read_csv(self.credents.networkFactorsLocation)

        self.currUniverse = list(set(self.prices[self.prices['date'] == max(self.prices['date'])]['symbol']))
        self.currUniverse.sort()

        self.currFactors = list(set(self.factors['symbol']))
        self.prices = self.prices[self.prices['symbol'].isin(self.currUniverse)]
        self.features = pd.concat([self.prices,self.factors], axis = 0).pivot(index='date',columns='symbol',values='value').fillna(method='ffill').fillna(value=0)


    def trainNetwork(self):
        print("Training network")
        self.trained = True
        self.mlPipe = mlPipeline(features=self.features.copy(), currentUniverse=self.currUniverse, windowLength=self.windowLength)


    def checkTrainStatus(self, force = False):
        self.updateData()

        if (self.currUniverse != self.universeCols) or (self.currFactors != self.factorCols):
            self.trainNetwork()
            self.universeCols = self.currUniverse
            self.factorCols = self.currFactors

        elif (date.today().weekday() == 6):
            self.trainNetwork()
            self.universeCols = self.currUniverse
            self.factorCols = self.currFactors
        elif (force):
            self.trainNetwork()
            self.universeCols = self.currUniverse
            self.factorCols = self.currFactors
        else:
            print("Does not need training")

    def controlNetwork(self):

        lastUpdateTrain = ''
        lastUpdatePredict = ''
        while True:
            if self.TimeRules.getTiming(lastUpdateTrain, ['deepLearning', 'checkTrainStatus']):
                lastUpdateTrain = date.today().strftime('%Y-%m-%d')
                self.checkTrainStatus()
            if self.TimeRules.getTiming(lastUpdatePredict, ['deepLearning','predict']):
                lastUpdatePredict = date.today().strftime('%Y-%m-%d')
                if not self.trained:
                    self.checkTrainStatus(force=True)
                else:
                    self.updateData()
                predictions = self.mlPipe.predict(features=self.features.copy())
                predictions.to_csv(self.credents.networkPredictionsLocation, index=False)

            else:
                time.sleep(self.credents.sleepSeconds)


class mlPipeline():
    def __init__(self, features, currentUniverse, splitPercents = [.88, .94, 1], windowLength = 252):

        self.splitPercents = splitPercents
        self.features = features
        self.currentUniverse = currentUniverse
        self.windowLength = windowLength


        self.numObs = len(self.features.index)
        self.trainSplit = int(self.numObs*self.splitPercents[0])
        self.valSplit = int(self.numObs*self.splitPercents[1])
        self.changes = [1, 5, 20, 252]
        self.featureMaxes = {}
        self.engineerFeatures()
        self.split()
        self.normalizePredicting()
        self.generateWindow()
        self.compile_and_fit()

    def engineerFeatures(self, usePrev = False):

        for col in self.features.columns:

            for change in self.changes:
                self.features[col + '_' + str(change)] = self.features[col].pct_change(change)
            if usePrev:
                maxDivider = self.featureMaxes[col]
            else:
                maxDivider = max(self.features[:self.trainSplit][col])
                self.featureMaxes[col] = max(self.features[:self.trainSplit][col])

            self.features[col + 'percentOfMax'] = self.features[col] / maxDivider

            self.features.drop(col, inplace=True, axis = 1)
        self.features.replace([np.inf, -np.inf], 0, inplace=True)
        self.features.fillna(value=0, inplace=True)

    def split(self):

        self.train_df, self.val_df, self.test_df = self.features[:self.trainSplit], self.features[self.trainSplit:self.valSplit], self.features[self.valSplit:]

    def normalizePredicting(self):
        self.train_mean = (self.train_df).min()
        self.train_std = self.train_df.max()

        self.train_df = (self.train_df - self.train_mean)/ (self.train_std - self.train_mean)
        self.val_df = (self.val_df - self.train_mean) / (self.train_std - self.train_mean)
        self.test_df = (self.test_df - self.train_mean) / (self.train_std - self.train_mean)


    def generateWindow(self):
        self.labels = [col + '_' + str(self.changes[0]) for col in self.currentUniverse]
        self.window = windowGenerator(self.windowLength, label_width = 1, shift = 1,
                                train_df=self.train_df, val_df = self.val_df, test_df = self.test_df,
                                label_columns = self.labels)


    def compile_and_fit(self):

        self.rnn_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(100, return_sequences = True),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.LSTM(150, return_sequences = True),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.LSTM(150, return_sequences = True),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dense(units = len(self.labels))
        ])

        MAX_EPOCHS = 10
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                         patience = 5,
                                                         mode = 'min')
        self.rnn_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                     optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005),
                     metrics = [tf.keras.metrics.MeanAbsoluteError()])




        history = self.rnn_model.fit(self.window.train, epochs = MAX_EPOCHS,
                           validation_data = self.window.val,
                           callbacks =[early_stopping])

        #self.window.plot(model=self.rnn_model, plot_col='AAPL_1')


    def predict(self, features):
        print('prediction')
        self.features = features

        self.engineerFeatures(usePrev=True)

        self.features = (self.features - self.train_mean)/(self.train_std - self.train_mean)
        print(self.features.tail(10))

        data = tf.expand_dims(tf.constant(self.features[-self.windowLength:]), axis = 0)

        preds = tf.squeeze(self.rnn_model((data))).numpy()

        dfObj = {self.labels[i].split('_')[0] : preds[i] for i in range(len(preds))}
        predictions = pd.DataFrame.from_records([dfObj])
        predictions['date'] = date.today().strftime('%Y-%m-%d')
        predictions = pd.melt(predictions, id_vars=['date'], var_name='symbol')
        return predictions












'''
windowGenerator
    A class to generate neural network windows
'''




class windowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df, val_df, test_df,
                label_columns = None):
        # Store the raw data

        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df

        # Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}


        # Work out the window parameters

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width

        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]


    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])

    def split_window(self, features):
        # Features is a 3d array, or tensor,
            # created with tf.keras.utils.timeseries_dataset_from_array function

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:, self.column_indices[name]] for name in self.label_columns],
                axis = -1)

        ## Slicing doesn't preserve static shape info, so set the shapes
        ## Manually. This way the tf.data.Datasets are easier to inspect.

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='AAPL_1', max_subplots = 3):
        inputs, labels = self.example
        plt.figure(figsize = (12, 8))
        plot_col_index = self.column_indices[plot_col]

        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col}[normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label = 'Inputs', marker = '.', zorder = -10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                       edgecolors = 'k', label = 'Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions = model(inputs)

                plt.scatter(self.label_indices, predictions[n, label_col_index],
                           marker = 'X', edgecolors = 'k', label = 'Predictions',
                           c = '#ff7f0e', s=64)
            if n==0:
                plt.legend()
        plt.xlabel('Time [h]')
        plt.savefig('plt.png')

    def make_dataset(self, data):

        data = np.array(data, dtype=np.float32)

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data = data,
            targets = None,
            sequence_length = self.total_window_size,
            sequence_stride = 1,
            shuffle= True,
            batch_size = 32,)
        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        '''Get and cache an esample batch of inputs, labels for plotting'''

        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found so get one from .train dataset
            result = next(iter(self.test))
            # and cache it for next time
            self._example = result
        return result
