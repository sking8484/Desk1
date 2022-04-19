from DataHub.dataHub import dataHub
from ion import ion

class ella:
    def __init__(self):
        pass
    def runDataHub(self):
        datahub = dataHub()
        datahub.maintainUniverse([15,0,0],[17,0,0])

controller = ella()
controller.runDataHub()