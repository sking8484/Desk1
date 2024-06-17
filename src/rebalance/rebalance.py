import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pandas as pd
from db_link.db_link import DataLink
import json
import threading
from abstract_classes_rebalance import Broker, OrderCreator
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class AlpacaLink(Broker):
    def __init__(self, accountDict: dict):
        self.accountObj = accountDict
        self.brokerApi = self.initializeBroker()
        
    def initializeBroker(self):
        alpaca_pubkey = self.accountObj['alpaca_pubkey']
        alpaca_seckey = self.accountObj['alpaca_seckey']

        return TradingClient(alpaca_pubkey, alpaca_seckey, paper=True)

    def initializeTestBroker(self, broker):
        self.brokerApi = broker

    def getBrokerApi(self):
        return self.brokerApi

    def getOpenPositions(self):
        return self.getBrokerApi().get_all_positions()
 
    def closeOpenOrders(self):
        self.getBrokerApi().cancel_orders()

    def getBuyingPower(self):
        return float(self.getBrokerApi().get_account().equity)
        
    def getOpenPosition(self, position: str):
        return self.getBrokerApi().get_open_position(position.upper())

    def getOpenPositionMarketValue(self, position: str):
        return self.getOpenPosition(position).market_value
        
    def placeTrade(self, order: dict):
        return self.getBrokerApi().submit_order(order)
        
    def liquidate(self, position: str):
        return self.getBrokerApi().close_position(position)


class AlpacaOrderCreator(OrderCreator):
    def __init__(self, dataLink, broker):
        self.dataLink = dataLink 
        self.broker = broker
        self.finalOrders = []

    def retrieveDesiredWeights(self):
        desiredWeights = self.dataLink.return_table(os.environ["MAINWEIGHTSTABLE"])
        desiredWeights = desiredWeights[desiredWeights['date'] == max(desiredWeights['date'])]
        print(desiredWeights)
        desiredWeights = json.loads(desiredWeights.to_json(orient='records'))
        return desiredWeights 

    def createUniverse(self, desiredWeights):
        universe = [pos['symbol'].upper() for pos in desiredWeights] 
        return universe

    def getOpenPositions(self):
        openPositionsList = [pos.symbol for pos in self.broker.getOpenPositions()] 
        return openPositionsList

    def createLiquidations(self, universe):
        liquidations = []
        for pos in self.broker.getOpenPositions():
            if not pos.symbol in universe:
                liquidations.append({'symbol':pos.symbol,'orderType':'LIQ'}) 
        return liquidations

    def createFinalOrders(self, desiredWeights, liquidations):
        self.finalOrders = []
        orders = []
        for currOrder in desiredWeights:
            if round(float(currOrder['value']),2) == 0:
                currOrder['orderType']="LIQ"
                currOrder['marketVal'] = 0
            else:
                currOrder['marketVal'] = round((self.broker.getBuyingPower())*float(currOrder['value']),2)
                currOrder['orderType']='TRADE'
                print(self.getOpenPositions())
                if currOrder['symbol'].upper() in self.getOpenPositions():
                    currOrder['marketVal'] -= float(self.broker.getOpenPositionMarketValue(currOrder['symbol'].upper()))
            currOrder['symbol'] = currOrder['symbol'].upper()
            orders.append(currOrder)
        self.finalOrders += liquidations
        orders.sort(key=lambda x: x["marketVal"])
        self.finalOrders += orders

        
    def buildOrderBook(self):
        desiredWeights = self.retrieveDesiredWeights()
        universe = self.createUniverse(desiredWeights)
        openPositions = self.getOpenPositions()
        liquidations = self.createLiquidations(universe)
        self.createFinalOrders(desiredWeights, liquidations)
        return self.finalOrders

class Rebalance:
    def __init__(self, broker: Broker):
        self.broker = broker
       
    def placeTrades(self, finalOrders):
        orders = []
        for order in finalOrders:
            time.sleep(1)
            if (order['orderType'] == 'LIQ'):
                try:
                    self.broker.liquidate(order['symbol'])
                    orders.append({"symbol":order['symbol'], "notional":"LIQ"})
                except Exception as e:
                    print(e)
            else:
                if order['marketVal'] > 0:
                    side = OrderSide.BUY
                else:
                    side = OrderSide.SELL
                try:
                    orderObj = {
                        "symbol":order['symbol'],
                        "notional":abs(round(order['marketVal'],2)),
                        "side":side,
                        "type":'market',
                        "time_in_force":'day'
                    }

                    marketOrderRequest = MarketOrderRequest(
                        symbol=orderObj["symbol"],
                        notional=orderObj["notional"],
                        side=orderObj["side"],
                        time_in_force=TimeInForce.DAY
                    )
                    self.broker.placeTrade(marketOrderRequest)
                    orders.append({"symbol":order['symbol'], "notional":abs(order['marketVal'])})
                except Exception as e:
                    print (e)
        return orders
