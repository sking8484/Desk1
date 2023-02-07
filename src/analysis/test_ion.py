import unittest
from analysis import ion 
import pandas as pd
import numpy as np
from analysis.ion import AnalysisMethods, GerberStatistic
from numpy import transpose as t
from sklearn.decomposition import PCA

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

    def test_calculate_svd(self):
        methods = AnalysisMethods()
        matrix = np.array([
                                [1, 2, 3],
                                [3, 2, 6],
                                [7, 24, 2]
                            ])

        svd_proc = methods.calculate_svd(matrix)
        self.assertTrue(np.allclose(matrix, svd_proc['elementary_matrices'].sum(axis=0), atol=1e-10))

    def test_filter_svd_matrices(self):
        
        methods = AnalysisMethods()
        matrix = np.array([
                                [1, 2, 3],
                                [3, 2, 6],
                                [7, 24, 2]
                            ])

        svd_proc = methods.calculate_svd(matrix)
        filtered_matrix = methods.filter_svd_matrices(svd_proc['elementary_matrices'], svd_proc['singular_values'], 95)
        expected = svd_proc['elementary_matrices'][0] + svd_proc['elementary_matrices'][1]
        self.assertEqual(filtered_matrix.tolist(), expected.tolist())


class TestGerberStatistic(unittest.TestCase):

    pandas_data = pd.DataFrame(
        np.array([
                     [0, 1, 2],
                     [3, 4, 5],
                     [6, 7, 8]
                 ]),
        columns=['a', 'b', 'c'])

    data = np.arange(9.0).reshape((3, 3))
    Q = .5

    def test_init_gerber(self):
        
        gerber_method = GerberStatistic(self.pandas_data, self.Q)
        self.assertIsInstance(gerber_method, GerberStatistic)

    def test_calculate_limits(self):
        
        gerber_method = GerberStatistic(self.pandas_data, self.Q)
        limits = gerber_method.calculate_limits(self.data, self.Q)
        self.assertEqual(limits['upperLimit'].tolist(), (self.Q*np.std(self.data, axis=0)).tolist())
        self.assertEqual(limits['lowerLimit'].tolist(), (-1*self.Q*np.std(self.data, axis=0)).tolist())

    def test_init_up_low_mats(self):
        upperLimit = np.array([3, 2, 1])
        lowerLimit = -1*upperLimit
        gerber_method = GerberStatistic(self.pandas_data, self.Q)
        matrices = gerber_method.initialize_upper_lower_matrices(self.data, upperLimit, lowerLimit)

        expected_pos = np.array([
                                    [-3, -1, 1],
                                    [0, 2, 4],
                                    [3, 5, 7]
                            ])
        expected_neg = np.array([
                                    [3, 3, 3],
                                    [6, 6, 6],
                                    [9, 9, 9]
                                ])
        self.assertEqual(matrices['upperMatrix'].tolist(), expected_pos.tolist())
        self.assertEqual(matrices['lowerMatrix'].tolist(), expected_neg.tolist())

    def test_calculate_upper_lower(self):
        upper_mat = np.array([
                                [-3, -1, 1],
                                [0, 2, 4],
                                [3, 5, 7]
                            ])
        lower_mat = np.array([
                                    [-1, 3, 3],
                                    [6, -2, 6],
                                    [9, 9, 9]
                                ])
        upper_stat = np.array([
                                  [0, 0, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]
                              ])

        lower_stat = np.array([
                                  [1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]
                              ])
        gerber_method = GerberStatistic(self.pandas_data, self.Q)
        matrices = gerber_method.calculate_upper_lower_matrices(upper_mat, lower_mat)
        self.assertEqual(matrices['upperMatrix'].tolist(), upper_stat.tolist())
        self.assertEqual(matrices['lowerMatrix'].tolist(), lower_stat.tolist())

    def test_calculate_mid_matrix(self):
        upper_stat = np.array([
                                  [0, 0, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]
                              ])

        lower_stat = np.array([
                                  [1, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]
                              ])
        gerber_method = GerberStatistic(self.pandas_data, self.Q)
        mid_matrix = gerber_method.calculate_mid_matrix(upper_stat, lower_stat)

        expected = np.array([
                                [0, 1, 0],
                                [0, 0, 0],
                                [0, 0, 0]
                            ])
        self.assertEqual(mid_matrix.tolist(), expected.tolist())

    def test_build_gerber_numerator(self):
        upper_stat = np.array([
                                  [0, 0, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]
                              ])

        lower_stat = np.array([
                                  [1, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]
                              ])
        N_UU = t(upper_stat) @ upper_stat
        N_DD = t(lower_stat) @ lower_stat
        N_UD = t(upper_stat) @ lower_stat
        N_DU = t(lower_stat) @ upper_stat

        gerber_method = GerberStatistic(self.pandas_data, self.Q)
        gerber_num = gerber_method.build_gerber_numerator(upper_stat, lower_stat)
        #self.assertEquals(gerber_num.tolist(), (N_UU + N_DD - N_UD - N_DU).tolist())

    def test_build_gerber_denom(self):
        gerber_method = GerberStatistic(self.pandas_data, self.Q)
        mid_matrix = np.array([
                                [0, 1, 0],
                                [0, 0, 0],
                                [0, 0, 0]
                            ])

        num_rows = 3
        gerber_denom = gerber_method.build_gerber_denominator(mid_matrix, num_rows)
        expected = np.array([
                                [3, 3, 3],
                                [3, 2, 3],
                                [3, 3, 3]
                            ])
        self.assertEquals(gerber_denom.tolist(), expected.tolist())

    def test_divide_gerber_matrices(self):
        numer = np.array([
                             [3, 6, 3],
                             [3, 4, 3],
                             [9, 12, 15]
                         ])
        denom = np.array([
                                [3, 3, 3],
                                [3, 2, 3],
                                [3, 3, 3]
                            ])
        gerber_method = GerberStatistic(self.pandas_data, self.Q)
        gerber_matrix = gerber_method.divide_gerber_matrices(numer, denom)
        expected = np.array([
                                [1, 2, 1],
                                [1, 2, 1],
                                [3, 4, 5]
                            ])
        self.assertEquals(gerber_matrix.tolist(),expected.tolist())

    def test_create_gerber_stat(self):
        gerber_method = GerberStatistic(self.pandas_data, self.Q)
        matrix = np.array([
                                [1, 2, 1],
                                [1, 2, 1],
                                [3, 4, 5]
                            ])
        methods = AnalysisMethods()
        diagonalized_matrix = methods.diagonalize_matrix(methods.calculate_std(self.data))
        gerber_stat = gerber_method.create_gerber_stat(diagonalized_matrix, matrix)
        #self.assertEquals(gerber_stat.tolist(), (diagonalized_matrix @ matrix @ diagonalized_matrix).tolist())


    def test_get_gerber_statistic(self):
        data = pd.read_csv('src/analysis/^XAU.csv')[['XAU Close', 'SPX CLOSE']]
        #data = pd.DataFrame(np.array([
        #                    [-1, 0],
        #                    [0, 1],
        #                    [1, -1]
        #                ]))
        gerber_method = GerberStatistic(data, .9)

        stat = gerber_method.get_gerber_statistic()


if __name__ == '__main__':
    unittest.main()
