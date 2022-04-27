import os 
import sys 
fpath = os.path.join(os.path.dirname(__file__),'DataHub')
sys.path.append(fpath)

from DataHub.dataHub import dataHub
from DataHub.dataLink import dataLink
from ion import ion
from DataHub.privateKeys.privateData import credentials

class ella:
    def __init__(self):
        self.credents = credentials()

        pass
    def runDataHub(self):
        datahub = dataHub()
        datahub.maintainUniverse([1,30,0],[2,0,0])
    def findEconomicConditions(self):
        pass

    def findOptimalWeights(self):
        datalink = dataLink(self.credents.credentials)
        data = datalink.returnTable(self.credents.mainStockTable)
        ionLink = ion(data)
        ionLink.getOptimalWeights()


controller = ella()
controller.findOptimalWeights()
