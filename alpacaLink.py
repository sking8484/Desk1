import alpaca_trade_api as tradeapi
import time
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
from DataHub.privateKeys.privateData import credentials
from DataHub.dataLink import dataLink
import json
import threading


class alpacaLink:
    def __init__(self):
        self.credents = credentials()

    def initAlpaca(self, accountObj):
        alpaca_pubkey = accountObj['alpaca_pubkey']
        alpaca_seckey = accountObj['alpaca_seckey']
        alpaca_baseurl = accountObj['alpaca_baseurl']

        self.alpaca = tradeapi.REST(alpaca_pubkey,alpaca_seckey,alpaca_baseurl,'v2')

    def initDataLink(self):
        self.dataLink = dataLink(self.credents.credentials)

    def getOpenPosCloseOpenOrders(self):
        self.openPositions = self.alpaca.list_positions()
        self.alpaca.cancel_all_orders()

    def getBuyingPower(self):
        account = self.alpaca.get_account()
        self.buying_power = float(account.equity)

    def rebalance(self, accountObj):

        self.initAlpaca(accountObj)
        self.getOpenPosCloseOpenOrders()
        self.initDataLink()
        self.getBuyingPower()


        ordersToSubmit = self.dataLink.returnTable(self.credents.weightsTable)
        ordersToSubmit = ordersToSubmit[ordersToSubmit['date'] == max(ordersToSubmit['date'])]
        ordersToSubmit = json.loads(ordersToSubmit.to_json(orient='records'))

        self.positionsToTrade = [pos['symbol'].upper() for pos in ordersToSubmit]
        self.openPositionsList = [pos.symbol for pos in self.openPositions]
        self.finalOrders = []
        self.sellNonUniverse()
        self.finalOrders += [self.createTrades(order) for order in ordersToSubmit]

        self.placeTrades(self.finalOrders)

    def sellNonUniverse(self):
        for pos in self.openPositions:
            if not pos.symbol in self.positionsToTrade:
                self.finalOrders.append({'symbol':pos.symbol,'orderType':'LIQ'})

    def createTrades(self, currOrder):
        if round(float(currOrder['value']),2) == 0:
            currOrder['orderType']="LIQ"
        else:
            currOrder['marketVal'] = round((self.buying_power)*float(currOrder['value']),2)
            currOrder['orderType']='TRADE'
            if currOrder['symbol'].upper() in self.openPositionsList:
                currOrder['marketVal'] -= float(self.alpaca.get_position(currOrder['symbol'].upper()).market_value)
        currOrder['symbol'] = currOrder['symbol'].upper()
        return currOrder

    def placeTrades(self, finalOrders):
        for order in finalOrders:
            print(order)
            time.sleep(1)
            if (order['orderType'] == 'LIQ'):
                try:
                    self.alpaca.close_position(order['symbol'])
                except Exception as e:
                    print(e)
            else:
                if order['marketVal'] > 0:
                    side = 'buy'
                else:
                    side = 'sell'
                try:
                    self.alpaca.submit_order(
                    symbol = order['symbol'],
                    notional = abs(order['marketVal']),
                        side = side,
                        type = 'market',
                        time_in_force='day'
                    )
                except Exception as e:
                    print (e)



def threadRebalance():
    alpacaCredents = credentials().alpaca_credents
    for i in range(len(alpacaCredents)):
        currAccount = alpacaCredents[i]
        x = threading.Thread(target=alpacaLink().rebalance, args=(currAccount,)).start()