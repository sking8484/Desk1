import unittest 
import numpy as np
from unittest.mock import patch
import pandas as pd
import alpaca.data as apdata
from datahub.dataHub import dataHub
from reportingsuite.reportingSuite  import reportingSuite

class TestReportingSuite(unittest.TestCase):
    def testSwitchTimes(self):
        suite = reportingSuite()
        input = [pd.to_datetime("2015-01-01 10:15:32"), pd.to_datetime("2015-01-02 10:15:32"), pd.to_datetime("2015-01-02 11:15:32")]
        output = suite.switchTimesToDates(input)

        expected = [pd.to_datetime("2015-01-01 00:00:00"), pd.to_datetime("2015-01-02 10:15:32"), pd.to_datetime("2015-01-02 00:00:00")]
        self.assertEqual(output, expected)
if __name__=="__main__":
    unittest.main()

