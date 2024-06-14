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
import boto3
import datetime


def handler(event, context):
    db_link = DataLink()
    load_dotenv()
    runMSSA(db_link)
    runGerber(db_link)
    runOptimization(db_link)
    env = os.environ["ENV"]
    sendMessage()
    return {'statusCode': 200}


def runMSSA(link):
    data = link.return_table(os.environ["MAINSTOCKTABLE"]).pivot(index = "date", columns = "symbol", values = "value")
    ssa = SpectrumAnalysis(data, L = 5, useIntercept = False, informationThreshold = .90, lookBack = 100)
    prediction = ssa.run_mssa()
    table = os.environ["MAINPREDICTIONTABLE"]

    link.append(table, prediction)


def runGerber(link):
    data = link.return_table(os.environ["MAINSTOCKTABLE"]).pivot(index = "date", columns = "symbol", values = "value")
    data_pctchange = data.apply(pd.to_numeric).pct_change().dropna()
    gerber = GerberStatistic(data_pctchange, .5)
    gerber_data = gerber.get_gerber_statistic().reset_index()
    gerber_data.rename(columns={"symbol":"index"}, inplace=True)
    gerber_data = gerber_data.melt(id_vars=["index"]).rename(columns = {"index":"symbol_1", "symbol":"symbol_2"})
    gerber_data["date"] = datetime.datetime.now()

    table = os.environ["MAINGERBERTABLE"]
    link.append(table, gerber_data)

def runOptimization(link):
    stock_data = link.return_table(os.environ["MAINSTOCKTABLE"]).pivot(index = "date", columns = "symbol", values = "value").reset_index()
    predictions = link.return_table(os.environ["MAINPREDICTIONTABLE"])
    predictions = predictions[predictions['date'] == max(predictions['date'])]
    optimizer = ion.ion()
    weights = optimizer.getOptimalWeights(stock_data, 50, 1.1, True, predictions,True)
    table = os.environ["MAINWEIGHTSTABLE"]
    link.append(table, weights)


def sendMessage():
    if os.environ["ENV"] != "local":
        sqs_client = boto3.client("sqs")
        queueUrl = sqs_client.get_queue_url(QueueName="rebalance-queue")["QueueUrl"]
        sqs_client.send_message(QueueUrl=queueUrl, MessageBody="Hi")
