import sys
import os
from dataHub import dataHub
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_link.db_link import DataLink

def handler():
    hub = dataHub(DataLink)

handler()
    
