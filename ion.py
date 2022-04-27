
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import transpose as t
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
import datetime   
import math

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

    def getOptimalWeights(self, data:pd.DataFrame, delta:Optional[float] = 50, leverageAmt: Optional[float] = 1.0, gerber:Optional[bool] = True) -> pd.DataFrame:
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
        cleaned_data = data.pct_change().dropna()
        N = len(cleaned_data.columns)

        comovement = matrix(self.getGerberMatrix(cleaned_data).values)
        returns = matrix(np.reshape(cleaned_data.mean().values,(N,1)))
        
        G = matrix(0.0,(N,N))
        G[::N+1] = -1.0
        h = matrix(0.0,(N,1))
        A = matrix(1.0,(1,N))
        b = matrix(leverageAmt)

        weights = qp(delta*comovement,-returns, G,h,A,b)['x']
        weights = np.floor(weights*1000)/1000

        weights = pd.DataFrame(weights, columns = ['weights'])
        weights['date'] = datetime.datetime.today().strftime("%Y-%m-%d")
        weights['Ticker'] = cleaned_data.columns
        return weights

        



        
        
        
        


    
        


        

        
        



        






    




