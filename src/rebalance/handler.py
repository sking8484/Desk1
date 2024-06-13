import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_link.db_link import test_setup
from db_link.db_link import DataLink
from datahub.dataHub import dataHub
import os
from dotenv import load_dotenv
from rebalance.rebalance import AlpacaLink, AlpacaOrderCreator, Rebalance

def handler(req, resp):
    db_link = DataLink()
    load_dotenv()
    alpaca_pubkey = os.environ["ALPACAPUBKEY"]
    alpaca_privkey = os.environ["ALPACAPRIVKEY"]
    credentsDict = {
        "alpaca_pubkey":alpaca_pubkey,
        "alpaca_seckey":alpaca_privkey
    }
    alpacaLink = AlpacaLink(credentsDict)
    orderCreator = AlpacaOrderCreator(db_link, alpacaLink)
    orders = orderCreator.buildOrderBook()
    rebalance = Rebalance(alpacaLink)
    rebalance.placeTrades(orders)

    return
