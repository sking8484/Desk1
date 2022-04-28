
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

    





    




