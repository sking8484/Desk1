import unittest
from rebalance.rebalance import Rebalance, AlpacaLink, AlpacaOrderCreator 

class Position:
    def __init__(self, symbol):
        self.symbol = symbol

class SpoofBrokerApi:
    def close_position(self, position):
        print(f"Closing Position {position}")

        return position

    def list_positions(self):
        return [Position("AAPL"), Position("TSLA")]

    def cancel_all_orders(self):
        print("canceling orders!")

    def get_account(self):
        class Account:
            def __init__(self):
                self.equity = 1000

        return Account()

    def get_position(self, position):
        return Position("GE")

    def submit_order(self, symbol,time_in_force, side, type, notional):
        print(f"{side} order for {symbol} created: ${notional}")

    def close_position(self, position):
        print(f"Closed position {position}")

def spoofWeightsData():
    return [
        {"date":"12-31-2021", "symbol":"AAPL", "value":".8"},
        {"date":"12-31-2021", "symbol":"TSLA", "value":".8"}
    ]

class TestRebalance(unittest.TestCase):

    def test_init(self):
        brokerLink = AlpacaLink({})    
        orderCreator = AlpacaOrderCreator(None, brokerLink)
        rebalance = Rebalance(brokerLink)

        self.assertEqual(brokerLink, rebalance.broker)

    def test_rebalance(self):
        brokerLink = AlpacaLink({
                                         "alpaca_pubkey":"fake",
                                         "alpaca_seckey":"fake",
                                         "alpaca_baseurl":"fake"
                                     })
        orderCreator = AlpacaOrderCreator(None, brokerLink)
        orderCreator.retrieveDesiredWeights = spoofWeightsData 
        spoofApi = SpoofBrokerApi()
        brokerLink.initializeTestBroker(spoofApi)
        orders = orderCreator.buildOrderBook()
        rebalance = Rebalance(brokerLink)
        orderLog = rebalance.placeTrades(orders)

        self.assertListEqual(orderLog, [{"symbol":"AAPL", "notional":800.0}, {"symbol":"TSLA", "notional":800.0}])

if __name__ == '__main__':
    unittest.main()
