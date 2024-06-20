import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataHub import dataHub
from db_link.db_link import DataLink
import urllib.request
import boto3

def handler(req, resp):
    hub = dataHub(DataLink)
    hub.maintainUniverse()
    hub.maintainFactors()
    sendMessage()
    return

def sendMessage():
    if os.environ["ENV"] != "local":
        sqs_client = boto3.client("sqs")
        queueUrl = sqs_client.get_queue_url(QueueName="analysis-queue")["QueueUrl"]
        sqs_client.send_message(QueueUrl=queueUrl, MessageBody="Hi")

