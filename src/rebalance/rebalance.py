import alpaca_trade_api as tradeapi
import time
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
from datahub.privateKeys.privateData import credentials
from datahub.dataLink import dataLink
import json
import threading
from abc import ABC, abstractmethod 

class Broker(ABC):
    @abstractmethod
    def __init__(self, accountObj: dict):
        pass 
    
    @abstractmethod
    def initializeBroker(self):
        pass 
    
    @abstractmethod
    def getBrokerDict(self):
        pass

    @abstractmethod
    def getOpenPositions(self):
        pass 

    @abstractmethod
    def closeOpenOrders(self):
        pass 

    @abstractmethod
    def getBuyingPower(self):
        pass

    @abstractmethod
    def getOpenPosition(self, position: str):
        pass
    
    @abstractmethod
    def getOpenPositionMarketValue(self, position: str):
        pass

    @abstractmethod
    def placeTrade(self, orderDict: dict):
        pass
        
    @abstractmethod
    def liquidate(self, position: str):
        pass

class AlpacaLink(Broker):
    def __init__(self, accountDict: dict):
        self.accountObj = accountDict
        self.brokerObj = None

    def initializeBroker(self):
        alpaca_pubkey = self.accountObj['alpaca_pubkey']
        alpaca_seckey = self.accountObj['alpaca_seckey']
        alpaca_baseurl = self.accountObj['alpaca_baseurl']

        self.brokerObj= tradeapi.REST(alpaca_pubkey,alpaca_seckey,alpaca_baseurl,'v2')

    def getBrokerDict(self):
        return self.brokerObj

    def getOpenPositions(self):
        return self.getBrokerDict().list_positions()
 
    def closeOpenOrders(self):
        self.getBrokerDict().cancel_all_orders()

    def getBuyingPower(self):
        return float(self.getBrokerDict().get_account().equity)
        
    def getOpenPosition(self, position: str):
        return self.getBrokerDict().get_position(position.upper())

    def getOpenPositionMarketValue(self, position: str):
        return self.getOpenPosition(position).market_value
        
    def placeTrade(self, orderDict: dict):
        self.getBrokerDict().submit_order(**orderObj)
        
    def liquidate(self, position: str):
        self.getBrokerDict().close_position(position)

class Rebalance:
    def __init__(self, broker: Broker):
        self.credents = credentials()
        self.dataLink = None #dataLink(self.credents.credentials)
        self.broker = broker

    def getOrdersToSubmit(self):
        """Need to reimplement this using a weights object!!!!!!!!!!!!!"""
        """THis violates design principle"""
        ordersToSubmit = self.dataLink.returnTable(self.credents.weightsTable)
        ordersToSubmit = ordersToSubmit[ordersToSubmit['date'] == max(ordersToSubmit['date'])]
        ordersToSubmit = json.loads(ordersToSubmit.to_json(orient='records'))
        return ordersToSubmit

    def rebalance(self):
        ordersToSubmit = self.getOrdersToSubmit()
        positionsToTrade = [pos['symbol'].upper() for pos in ordersToSubmit]
        openPositionsList = [pos.symbol for pos in self.broker.getOpenPositions()]
        finalOrders = []
        sellNonUniverse(positionsToTrade, finalOrders)
        finalOrders += [self.createTrades(order) for order in ordersToSubmit]
        finalOrders = sorted(finalOrders, key = lambda d:d['marketVal'])
        placeTrades(finalOrders)

    def sellNonUniverse(self, positionsToTrade, finalOrders):
        for pos in self.broker.getOpenPositions():
            if not pos.symbol in positionsToTrade:
                finalOrders.append({'symbol':pos.symbol,'orderType':'LIQ'})

    def createTrades(self, currOrder):
        if round(float(currOrder['value']),2) == 0:
            currOrder['orderType']="LIQ"
            currOrder['marketVal'] = 0
        else:
            currOrder['marketVal'] = round((self.broker.getBuyingPower())*float(currOrder['value']),2)
            currOrder['orderType']='TRADE'
            if currOrder['symbol'].upper() in self.broker.getOpenPositions():
                currOrder['marketVal'] -= float(self.broker.getPositionMarketValue(currOrder['symbol'].upper()))
        currOrder['symbol'] = currOrder['symbol'].upper()
        return currOrder

    def placeTrades(self, finalOrders):
        for order in finalOrders:
            print(order)
            time.sleep(1)
            if (order['orderType'] == 'LIQ'):
                try:
                    self.broker.liquidate(order['symbol'])
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
                except Exception as e:
                    print (e)



def threadRebalance():
    alpacaCredents = credentials().alpaca_credents
    for i in range(len(alpacaCredents)):
        currAccount = alpacaCredents[i]
        x = threading.Thread(target=alpacaLink().rebalance, args=(currAccount,)).start()
