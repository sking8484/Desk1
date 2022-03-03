import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ion import ion
from numpy import transpose as t
filePath = "/Users/darstking/Desktop/Data/CMF/Finance/Trading/Desk1/testData/GerberTest.csv"

data = pd.read_csv(filePath,parse_dates=True)

analyzer = ion(data, Q = 1.1, delta = .1)

gerberWeights = analyzer.getOptimalWeights()
covWeights = analyzer.getOptimalWeights(gerber=False)
print(gerberWeights)