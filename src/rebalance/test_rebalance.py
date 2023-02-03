import unittest
from rebalance.rebalance import Rebalance, AlpacaLink, AlpacaOrderCreator 
import boto3
import json
import alpaca_trade_api

class Position:
    def __init__(self, symbol):
        self.symbol = symbol

class SpoofBrokerApi:
    def close_position(self, position):
        return position

    def list_positions(self):
        return [Position("AAPL"), Position("TSLA")]

    def cancel_all_orders(self):
        return []

    def get_account(self):
        class Account:
            def __init__(self):
                self.equity = 1000

        return Account()

    def get_position(self, position):
        return Position(position)

    def submit_order(self, symbol,time_in_force, side, type, notional):
        return {"symbol":symbol, "time_in_force":time_in_force, "side":side, "type":type, "notional":"notional"}

    def close_position(self, position):
        return position

def spoofWeightsData():
    return [
        {"date":"12-31-2021", "symbol":"AAPL", "value":".8"},
        {"date":"12-31-2021", "symbol":"TSLA", "value":".8"}
    ]

class TestBroker(unittest.TestCase):
    def get_credents(self):
        secret = boto3.client('secretsmanager')
        kwargs = {'SecretId': "alpaca_test_credents"}
        response = secret.get_secret_value(**kwargs)
        return json.loads(response['SecretString'])
        
    def test_init(self):
        credents = self.get_credents()
        broker = AlpacaLink(credents)
        broker.initializeBroker()

    def test_open_pos(self):
        credents = self.get_credents()
        broker = AlpacaLink(credents)
        broker.initializeBroker()
        positions = broker.getOpenPositions()
        self.assertIsInstance(positions, list)

    def test_close_positions(self):
        credents = self.get_credents()
        broker = AlpacaLink(credents)
        broker.initializeBroker()
        broker.closeOpenOrders()

    def test_get_buying_power(self):
        credents = self.get_credents()
        broker = AlpacaLink(credents)
        broker.initializeBroker()
        buyingPower = broker.getBuyingPower()
        self.assertGreaterEqual(round(buyingPower, 0), 90000)

    def test_get_open_position(self):
        credents = self.get_credents()
        broker = AlpacaLink(credents)
        broker.initializeBroker()
        position = broker.getOpenPosition("AAPL")
        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(round(float(position.qty), 0), (13))

    def test_place_trade(self):
        credents = self.get_credents()
        broker = AlpacaLink(credents)
        broker.initializeBroker()
        trade = {
            "symbol":"BRK.B",
            "side":"buy",
            "notional":"100",
            "type":"market",
            "time_in_force":"day"
        }
        order = broker.placeTrade(trade)
        self.assertEqual(order.symbol, "BRK.B")
        self.assertEqual(float(order.notional), float(100))

    def test_liquidate(self):
        self.test_place_trade()
        credents = self.get_credents()
        broker = AlpacaLink(credents)
        broker.initializeBroker()
        liquidation = broker.liquidate("BRK.B")
        self.assertEqual(liquidation.symbol, "BRK.B")

    def test_nonheld_liquidation(self):
        credents = self.get_credents()
        broker = AlpacaLink(credents)
        broker.initializeBroker()
        self.assertRaises(alpaca_trade_api.rest.APIError, broker.liquidate, "LUV")


class TestRebalance(unittest.TestCase):

    def test_init(self):
        brokerLink = AlpacaLink({})    
        orderCreator = AlpacaOrderCreator(None, brokerLink)
        rebalance = Rebalance(brokerLink)

        self.assertEqual(brokerLink, rebalance.broker)

    def test_buyonly_rebalance(self):
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

    def test_liquidation_rebalance(self):
        brokerLink = AlpacaLink({
                                         "alpaca_pubkey":"fake",
                                         "alpaca_seckey":"fake",
                                         "alpaca_baseurl":"fake"
                                     })
        orderCreator = AlpacaOrderCreator(None, brokerLink)
        orderCreator.retrieveDesiredWeights = spoofWeightsData 
        spoofApi = SpoofBrokerApi()
        def list_mult_pos():
            return [Position("GE"), Position("AAPL"), Position("TSLA")]
        spoofApi.list_positions = list_mult_pos
        brokerLink.initializeTestBroker(spoofApi)
        orders = orderCreator.buildOrderBook()
        rebalance = Rebalance(brokerLink)
        orderLog = rebalance.placeTrades(orders)
        self.assertEqual(orderLog, [{"symbol":"GE", "notional":"LIQ"}, {"symbol":"AAPL", "notional":800.0}, {"symbol":"TSLA", "notional":800.0}])
 

if __name__ == '__main__':
    unittest.main()
