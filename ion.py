
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import transpose as t
import sys
import datetime
from datetime import date
import math
from timeRules import TimeRules

from DataHub.privateKeys.privateData import credentials
import time

"""
The ion class is Desk1's main tool for analyzing data. This is where many of Desk1's trading signal's will be created.
"""


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

    def getGerberMatrix(self, data:pd.DataFrame, Q:Optional[float]=.5) -> pd.DataFrame:
        """
        A method to return the Gerber (modified COV) matrix using data
        ...

        Attributes
        ----------
        data: A numerical Pandas DataFrame (excludes dates, ID or anything of that variety)
            Data must be a percentage change dataframe.
            DO NOT INCLUDE DATE COLUMN

        Q: a fraction from (0,1]
            Indicating how many standard deviations
                We want to start counting comovements
        """



        data = data.copy()

        T = len(data)
        diag = np.diag(np.asarray(data.std()))

        upperLimit = Q*data.std()
        lowerLimit = -1*Q*data.std()

        upperMatrix = data - upperLimit
        lowerMatrix = data - lowerLimit

        upperMatrix = np.asmatrix(upperMatrix)
        lowerMatrix = np.asmatrix(lowerMatrix)

        upperMatrix[upperMatrix >= 0] = 1
        upperMatrix[upperMatrix < 0] = 0

        lowerMatrix[lowerMatrix <= 0] = -1
        lowerMatrix[lowerMatrix > 0] = 0
        lowerMatrix[lowerMatrix == -1] = 1

        """
        General idea of mid matrix is that if both
        upper and lower are 0 then it is a mid point
        but we need to add 1 to this because when we take dot product
        if we have a 1 and a 0 (above/below and inbetween) we get an inbetween
        this is incorrect
        """

        midMatrix = upperMatrix + lowerMatrix
        midMatrix = midMatrix + 1
        midMatrix[midMatrix == 2] = 0

        N_UU = t(upperMatrix) @ upperMatrix
        N_DD = t(lowerMatrix) @ lowerMatrix
        N_NN = t(midMatrix) @ midMatrix
        N_UD = t(upperMatrix) @ lowerMatrix
        N_DU = t(lowerMatrix) @ upperMatrix

        denom_mat = np.copy(N_NN)
        denom_mat[denom_mat > -100000] = T
        denom_mat = denom_mat - N_NN
        num_mat = N_UU + N_DD - N_UD - N_DU

        g = np.divide(num_mat, denom_mat)
        G = diag @ g @ diag

        return pd.DataFrame(G, index = data.columns, columns = data.columns)

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
        print(returns)

        G = matrix(0.0,(N,N))
        G[::N+1] = -1.0
        h = matrix(0.0,(N,1))
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
        self.credents = credentials()
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
        self.train_mean = (self.train_df).mean()
        self.train_std = self.train_df.std()

        self.train_df = (self.train_df - self.train_mean)/ self.train_std
        self.val_df = (self.val_df - self.train_mean) / self.train_std
        self.test_df = (self.test_df - self.train_mean) / self.train_std

    def generateWindow(self):
        self.labels = [col + '_' + str(self.changes[0]) for col in self.currentUniverse]
        self.window = windowGenerator(self.windowLength, label_width = 1, shift = 1,
                                train_df=self.train_df, val_df = self.val_df, test_df = self.test_df,
                                label_columns = self.labels)


    def compile_and_fit(self):

        self.rnn_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, return_sequences = True),
            tf.keras.layers.LSTM(90, return_sequences = True),
	        tf.keras.layers.LSTM(60, return_sequences = True),
            tf.keras.layers.LSTM(30),
            tf.keras.layers.Dense(units = len(self.labels))
        ])

        MAX_EPOCHS = 20
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                         patience = 5,
                                                         mode = 'min')
        self.rnn_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                     optimizer = tf.keras.optimizers.Adam(),
                     metrics = [tf.keras.metrics.MeanAbsoluteError()])




        history = self.rnn_model.fit(self.window.train, epochs = MAX_EPOCHS,
                           validation_data = self.window.val,
                           callbacks =[early_stopping])

        #self.window.plot(model=self.rnn_model, plot_col='AAPL_1')


    def predict(self, features):
        print('prediction')
        self.features = features

        self.engineerFeatures(usePrev=True)

        self.features = (self.features - self.train_mean)/self.train_std

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
