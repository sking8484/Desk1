import unittest
from analysis import ion 
import pandas as pd
import numpy as np
from analysis.ion import AnalysisMethods

class TestAnalysisMethods(unittest.TestCase):
    
    def test_calc_num_rows(self):
        methods = AnalysisMethods()
        x = np.array([1, 2, 3, 4])
        self.assertEqual(methods.calculate_num_rows(x), 4)

        x = np.zeros((2, 3, 4))
        self.assertEqual(methods.calculate_num_rows(x), 2)
            
    def test_diagonalize_matrix(self):
        methods = AnalysisMethods()
        x = np.array([1, 2])
        output = np.array([[1,0],[0,2]])
        self.assertEqual(methods.diagonalize_matrix(x).tolist(), output.tolist())

    def test_calculate_std(self):
        methods = AnalysisMethods()
        x = np.array([[1, 2], [3, 4]])
        self.assertEqual(methods.calculate_std(x).tolist(), np.array([1.,  1.]).tolist())

    def test_divide_matrices(self):
        methods = AnalysisMethods()
        x1 = np.arange(9.0).reshape((3, 3))
        x2 = 2 * np.ones(3)
        output = np.array([[0. , 0.5, 1. ],
               [1.5, 2. , 2.5],
               [3. , 3.5, 4. ]])
        self.assertEqual(methods.divide_matrices(x1, x2).tolist(), output.tolist())
if __name__ == '__main__':
    unittest.main()
