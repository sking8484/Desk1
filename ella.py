import os 
import sys 
fpath = os.path.join(os.path.dirname(__file__),'DataHub')
sys.path.append(fpath)

from DataHub.dataHub import dataHub
from ion import ion


class ella:
    def __init__(self):
        pass
    def runDataHub(self):
        datahub = dataHub()
        datahub.maintainUniverse([15,0,0],[19,40,0])

controller = ella()
controller.runDataHub()