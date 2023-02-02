import unittest
from analysis import ion 
import pandas as pd
import numpy as np

class TestIon(unittest.TestCase):
    
    def testGerberMatrix(self):
        testIon = ion.ion()
        df = pd.DataFrame(np.random.randint(0,100,size=(15, 4)), columns=list('ABCD'))
        output = testIon.getGerberMatrix(df)
        self.assertEqual(output.shape, (4, 4))
            
if __name__ == '__main__':
    unittest.main()
