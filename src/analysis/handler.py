import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_link.db_link import test_setup
from db_link.db_link import DataLink
from datahub.dataHub import dataHub
from ion import SpectrumAnalysis, GerberStatistic
from dotenv import load_dotenv
import ion
import pandas as pd

def handler():
    db_link = DataLink()
    #runMSSA(db_link)
    runGerber(db_link)

def runMSSA(link):
    load_dotenv()
    data = link.return_table(os.environ["MAIN_STOCK_TABLE"]).pivot(index = "date", columns = "symbol", values = "value")
    ssa = SpectrumAnalysis(data, L = 10, useIntercept = False, informationThreshold = .99, lookBack = 1000)
    prediction = ssa.run_mssa()
    table = os.environ["MAIN_PREDICTION_TABLE"]
    print(prediction)

    link.append(table, prediction)


def runGerber(link):
    data = link.return_table(os.environ["MAIN_STOCK_TABLE"]).pivot(index = "date", columns = "symbol", values = "value")
    data_pctchange = data.apply(pd.to_numeric).pct_change().dropna()
    gerber = GerberStatistic(data_pctchange, .5)
    print(gerber.get_gerber_statistic())

handler()


