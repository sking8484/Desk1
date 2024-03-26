import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_link.db_link import test_setup
from db_link.db_link import DataLink
from datahub.dataHub import dataHub
from ion import SpectrumAnalysis
from dotenv import load_dotenv
import ion

def handler():
    db_link = DataLink()
    runMSSA(db_link)

def runMSSA(link):
    load_dotenv()
    data = link.return_table(os.environ["MAIN_STOCK_TABLE"]).pivot(index = "date", columns = "symbol", values = "value")
    print(data)
    ssa = SpectrumAnalysis(data, L = 10, useIntercept = False, informationThreshold = .99, lookBack = 1000)
    prediction = ssa.run_mssa()
    print(prediction)

def runGerber():
    pass

handler()


