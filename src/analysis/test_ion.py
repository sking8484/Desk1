import unittest
from analysis import ion 
import pandas as pd
import numpy as np
from analysis.ion import AnalysisMethods, GerberStatistic, SpectrumAnalysis, QuantTools
from numpy import transpose as t
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import datetime

class TestQuantTools(unittest.TestCase):
    data = pd.DataFrame(data = np.array([np.arange(10), np.arange(10)]))
    Q = .5
    L = 5
    lookBack = 10
    def test_get_gerber(self):
        quanttools = QuantTools(data = self.data)
        self.assertIsInstance(quanttools.get_gerber(self.Q), GerberStatistic)

    def test_get_mssa(self):
        quanttools = QuantTools(data = self.data)
        self.assertIsInstance(quanttools.get_mssa(L = self.L), SpectrumAnalysis)

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
        filtered_matrix = methods.filter_svd_matrices(svd_proc['elementary_matrices'], svd_proc['singular_values'], .95)
        expected = svd_proc['elementary_matrices'][0] + svd_proc['elementary_matrices'][1]
        self.assertEqual(filtered_matrix.tolist(), expected.tolist())

    def test_run_regression(self):
        
        methods = AnalysisMethods()
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3 

        modelObj = methods.run_regression(X, y, False, True)
        self.assertEquals(np.round(modelObj.coefficients, 0).tolist(), np.array([1., 2.]).tolist())

    def test_clean_data(self):
        data = [
            [1, 2, 3],
            [4, 5, np.nan],
            [5, 7, 9]
        ]
        data = pd.DataFrame(data, columns = ['date', 'col1', 'col2'])
        methods = AnalysisMethods()
        expected_data = [
            [5],
            [7]
        ]
        expected = pd.DataFrame(expected_data, columns = ['col1'])
        cleaned_data = methods.clean_data(data, 2, removeNullCols = True, removeDateColumn = True)
        self.assertEquals(cleaned_data.columns, ['col1'])
        self.assertEquals(cleaned_data.to_numpy().tolist(), np.array(expected_data).tolist())

class TestSpectrumAnalysis(unittest.TestCase):
    
    singleColumn = np.arange(10)
    data = t(np.array([np.arange(10), np.arange(10), np.arange(10)]))
    df3 = pd.DataFrame(data, columns = ['col1', 'col2', 'col3'])

    def test_init(self):
        analysis = SpectrumAnalysis(self.df3, 2, 10)
        self.assertIsInstance(analysis, SpectrumAnalysis)

    def test_create_prediction_features(self):
        analysis = SpectrumAnalysis(self.df3, 5, 10)
        data = np.transpose(np.array([
            [1, 2, 3],
            [4, 5, 6]
                                     ]))
        input_data = np.column_stack([data, data])
        column_list = ['col_1', 'col_2']
        L = 3
        lookBack = 6

        prediction_features = analysis.create_prediction_features(input_data, column_list, L, lookBack)
        self.assertEquals(prediction_features['col_1'].tolist(), np.array([[5, 6]]).tolist())

    def test_create_page_matrix(self):
        
        analysis = SpectrumAnalysis(self.df3, 2, 10)
        matrix = analysis.create_page_matrix(self.singleColumn, 2, 10)
        expected = np.array([
            [0, 2, 4, 6, 8],
            [1, 3, 5, 7, 9]
        ])
        self.assertEquals(matrix.tolist(), expected.tolist())

    def test_concat_matrices(self):
        base = np.array([
                            [1, 2, 3],
                            [4, 5, 6]
                        ])
        additional = np.array([
                                  [7, 8, 9],
                                  [10, 11, 12]
                              ])

        expected = np.array([
                                [1, 2, 3, 7, 8, 9],
                                [4, 5, 6, 10, 11, 12]
                            ])

        analysis = SpectrumAnalysis(self.df3, 2, 10)
        concated = analysis.concat_matrices(base, additional)
        self.assertEquals(concated.tolist(), expected.tolist())

    def test_create_hsvt_matrix(self):
        analysis = SpectrumAnalysis(self.df3, 2, 10)
        hsvt_matrix = analysis.create_hsvt_matrix(self.df3, 2, 10)

        self.assertEquals(hsvt_matrix.shape, (2, 15))

    def test_create_labels_features(self):
        analysis = SpectrumAnalysis(self.df3, 2, 10)
        data = np.array([
                            [1, 3, 5],
                            [2, 4, 6],
                            [3, 5, 7]
                        ])
        labels_features = analysis.create_labels_features(data)
        expected_features = np.array([
                                       [1, 3, 5],
                                       [2, 4, 6]
                                   ])

        expected_labels = np.array([3, 5, 7])
        self.assertEquals(labels_features['labels'].tolist(), expected_labels.tolist())
        self.assertEquals(labels_features['features'].tolist(), expected_features.tolist())

    def test_learn_linear_model(self):
        analysis = SpectrumAnalysis(self.df3, 2, 10)
        data = np.array([
                            [1, 3, 5],
                            [2, 4, 6],
                            [3, 5, 7]
                        ])
        labels_features = analysis.create_labels_features(data)
        model = analysis.learn_linear_model(labels = labels_features['labels'], features = labels_features['features'])

    def test_predict(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3 
        expected = {
            'row1':np.array([16.]).tolist()
        }
        reg = LinearRegression().fit(X, y)
        predictors = {
            'row1':np.array([[3, 5]])
        }
        analysis = SpectrumAnalysis(self.df3, 2, 10)
        predictions = analysis.predict(reg, predictors)
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        self.assertEqual(np.round(predictions[today]['row1'].tolist(),0), np.round(expected['row1'],0))

    def test_run_mssa(self):
        data = pd.read_csv("src/analysis/prices.csv")[["Date", "AAPL", "TSLA", "MSFT"]]
        ssa = SpectrumAnalysis(data, L = 5, useIntercept = False, informationThreshold = .95, lookBack = 1000)
        prediction = ssa.run_mssa()

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
