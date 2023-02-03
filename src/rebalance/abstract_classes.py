from abc import ABC, abstractmethod 

class Broker(ABC):
    @abstractmethod
    def __init__(self, accountObj: dict):
        pass 
    
    @abstractmethod
    def initializeBroker(self, brokerApi):
        pass 
    
    @abstractmethod
    def getBrokerApi(self):
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

class OrderCreator(ABC):

    @abstractmethod
    def __init__(self, dataLink, broker):
        pass

    @abstractmethod
    def retrieveDesiredWeights(self):
        pass

    @abstractmethod
    def createUniverse(self, desiredWeights):
        pass

    @abstractmethod
    def getOpenPositions(self):
        pass

    @abstractmethod
    def createLiquidations(self, universe):
        pass

    @abstractmethod
    def createFinalOrders(self, desiredWeights, liquidations):
        pass 
    
    @abstractmethod
    def buildOrderBook(self):
        pass


