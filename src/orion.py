
import pandas as pd
import numpy as np
from typing import Optional
import datetime
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options




def getOptimalWeights(data:pd.DataFrame, delta:Optional[float] = 50, leverageAmt: Optional[float] = 1.0, gerber:Optional[bool] = True) -> pd.DataFrame:
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