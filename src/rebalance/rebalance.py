import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alpaca_trade_api as tradeapi
import time
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
from db_link.db_link import DataLink
import json
import threading
from abstract_classes_rebalance import Broker, OrderCreator

class AlpacaLink(Broker):
    def __init__(self, accountDict: dict):
        self.accountObj = accountDict
        self.brokerApi = "Main broker API. Requires call to initialize to be setup"
        
    def initializeBroker(self):
        alpaca_pubkey = self.accountObj['alpaca_pubkey']
        alpaca_seckey = self.accountObj['alpaca_seckey']

        if 'alpaca_baseurl' in self.accountObj.keys():
            alpaca_baseurl = self.accountObj['alpaca_baseurl']
        else:
            alpaca_baseurl = "https://paper-api.alpaca.markets"
            
        self.brokerApi= tradeapi.REST(alpaca_pubkey,alpaca_seckey,alpaca_baseurl,'v2')

    def initializeTestBroker(self, broker):
        self.brokerApi = broker

    def getBrokerApi(self):
        return self.brokerApi

    def getOpenPositions(self):
        return self.getBrokerApi().list_positions()
 
    def closeOpenOrders(self):
        self.getBrokerApi().cancel_all_orders()

    def getBuyingPower(self):
        return float(self.getBrokerApi().get_account().equity)
        
    def getOpenPosition(self, position: str):
        return self.getBrokerApi().get_position(position.upper())

    def getOpenPositionMarketValue(self, position: str):
        return self.getOpenPosition(position).market_value
        
    def placeTrade(self, order: dict):
        return self.getBrokerApi().submit_order(**order)
        
    def liquidate(self, position: str):
        return self.getBrokerApi().close_position(position)


class AlpacaOrderCreator(OrderCreator):
    def __init__(self, dataLink, broker):
        self.dataLink = dataLink 
        self.broker = broker
        self.finalOrders = []

    def retrieveDesiredWeights(self):
        desiredWeights = self.dataLink.returnTable(self.credents.weightsTable)
        desiredWeights = desiredWeights[desiredWeights['date'] == max(desiredWeights['date'])]
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
                if currOrder['symbol'].upper() in self.broker.getOpenPositions():
                    currOrder['marketVal'] -= float(self.broker.getPositionMarketValue(currOrder['symbol'].upper()))
            currOrder['symbol'] = currOrder['symbol'].upper()
            orders.append(currOrder)
        self.finalOrders += liquidations
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
                    side = 'buy'
                else:
                    side = 'sell'
                try:
                    orderObj = {
                        "symbol":order['symbol'],
                        "notional":abs(order['marketVal']),
                        "side":side,
                        "type":'market',
                        "time_in_force":'day'
                    }
                    self.broker.placeTrade(orderObj)
                    orders.append({"symbol":order['symbol'], "notional":abs(order['marketVal'])})
                except Exception as e:
                    print (e)
        return orders



def handler():
    alpacaCredents = credentials().alpaca_credents
    for i in range(len(alpacaCredents)):
        currAccount = alpacaCredents[i]
        x = threading.Thread(target=alpacaLink().rebalance, args=(currAccount,)).start()
