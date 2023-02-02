import unittest
from rebalance import rebalance 

class TestRebalance(unittest.TestCase):
    
    def test_init(self):
        alpacaLink = rebalance.AlpacaLink({})
        rebalObj = rebalance.Rebalance(alpacaLink)
        self.assertEqual(alpacaLink, rebalObj.broker)
    
if __name__ == '__main__':
    unittest.main()
