import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import transpose as t

"""
The ion class is Desk1's main tool for analyzing data. This is where many of Desk1's trading signal's will be created.
"""


class ion:
    def __init__(self, data, Q = .5, delta = .01):

        """
        A class used to represent a toolkit surrounding the Pandas dataframe data.
        ...

        Attributes
        ----------
        data: A pandas dataframe. 
            Always containing the dates NOT INDEXED as the left most column
            Always has dependent variable on the right most column
            Always has returns data [X_{t+1}/X_{t} - 1]
            N x M
        
        Q: The SD multiplier for gerberMatrix
            Defaults to 0.5

        Methods
        -------
        getGerberMatrix: Returns Numpy matrix 
            (gerberMatrix) 
            dimensions M x M

        """
        
        self.data = data
        self.numericalData = np.asmatrix(data.iloc[:,1:])
        self.Q = Q
        self.delta = delta
    
    def getGerberMatrix(self, Q = None):
        """
        A method to return the Gerber (modified COV) matrix using data
        ...

        Attributes
        ----------
        Q: a fraction from (0,1] 
            Indicating how many standard deviations 
                We want to start counting comovements
        """
        if Q == None:
            Q = self.Q

        
        data = self.data.iloc[:, 1:].copy()
                                                  
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

        return G 

    def getOptimalWeights(self, delta = None, gerber = True):
        """
        A method to get the optimal weights.
            Equation we are optimizing is r^T(h), w.r.t h^t(S)h = d^2
                r = returns vector
                h = weights (solving for)
                S = positive semidefinite comovement matrix
                d = variance of portfolio
        !!! Important, will be migrating to Orion shortly !!!
        """
        if delta == None:
            delta = self.delta

        returns = t(np.asmatrix(self.data.iloc[:, 1:].sum()))

        if gerber:
            cov_matrix = self.getGerberMatrix()
        else:
            cov_matrix = np.cov(self.numericalData, rowvar = False)
        
        
        
        cov_inv = np.linalg.inv(cov_matrix)

        weightNum = delta * (cov_inv @ returns)
        weightDenom = np.sqrt(t(returns) @ cov_inv @ returns)

        return t(np.divide(weightNum, weightDenom))
        
    
        


        

        
        



        






    




