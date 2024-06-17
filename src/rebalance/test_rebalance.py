import unittest
from rebalance.rebalance import Rebalance, AlpacaLink, AlpacaOrderCreator 
import json

class Position:
    def __init__(self, symbol):
        self.symbol = symbol
        self.market_value =0 

class SpoofBrokerApi:
    def close_position(self, position):
        return position

    def get_all_positions(self):
        return [Position("AAPL"), Position("TSLA")]

    def cancel_all_orders(self):
        return []

    def get_account(self):
        class Account:
            def __init__(self):
                self.equity = 1000

        return Account()

    def get_open_position(self, position):
        return Position(position)

    def submit_order(self, marketOrderRequest):
        return ""

    def close_position(self, position):
        return position

def spoofWeightsData():
    return [
        {"date":"12-31-2021", "symbol":"AAPL", "value":".8"},
        {"date":"12-31-2021", "symbol":"TSLA", "value":".8"}
    ]

class TestRebalance(unittest.TestCase):

    def test_init(self):
        brokerLink = AlpacaLink({
                                         "alpaca_pubkey":"fake",
                                         "alpaca_seckey":"fake"
                                     })    
        orderCreator = AlpacaOrderCreator(None, brokerLink)
        rebalance = Rebalance(brokerLink)

        self.assertEqual(brokerLink, rebalance.broker)

    def test_buyonly_rebalance(self):
        brokerLink = AlpacaLink({
                                         "alpaca_pubkey":"fake",
                                         "alpaca_seckey":"fake"
                                     })
        orderCreator = AlpacaOrderCreator(None, brokerLink)
        orderCreator.retrieveDesiredWeights = spoofWeightsData 
        spoofApi = SpoofBrokerApi()
        brokerLink.initializeTestBroker(spoofApi)
        orders = orderCreator.buildOrderBook()
        rebalance = Rebalance(brokerLink)
        orderLog = rebalance.placeTrades(orders)

        self.assertListEqual(orderLog, [{"symbol":"AAPL", "notional":800.0}, {"symbol":"TSLA", "notional":800.0}])

    def test_liquidation_rebalance(self):
        brokerLink = AlpacaLink({
                                         "alpaca_pubkey":"fake",
                                         "alpaca_seckey":"fake"
                                     })
        orderCreator = AlpacaOrderCreator(None, brokerLink)
        orderCreator.retrieveDesiredWeights = spoofWeightsData 
        spoofApi = SpoofBrokerApi()
        def list_mult_pos():
            return [Position("GE"), Position("AAPL"), Position("TSLA")]
        spoofApi.get_all_positions = list_mult_pos
        brokerLink.initializeTestBroker(spoofApi)
        orders = orderCreator.buildOrderBook()
        rebalance = Rebalance(brokerLink)
        orderLog = rebalance.placeTrades(orders)
        self.assertEqual(orderLog, [{"symbol":"GE", "notional":"LIQ"}, {"symbol":"AAPL", "notional":800.0}, {"symbol":"TSLA", "notional":800.0}])
 

if __name__ == '__main__':
    unittest.main()
