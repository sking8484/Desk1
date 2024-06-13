import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataHub import dataHub
from db_link.db_link import DataLink
import urllib.request

def handler(req, resp):
    hub = dataHub(DataLink)
    hub.maintainUniverse()
    return

