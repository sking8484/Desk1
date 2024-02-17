import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipaddress import collapse_addresses
import pandas as pd
import mysql.connector as sqlconnection
from inspect import trace
import numpy as np
import traceback
import abstract_classes_db_link
import logging


import mysql.connector
from mysql.connector import errorcode

DB_NAME = 'test'

TABLES = {}
TABLES['prices'] = (
    "CREATE TABLE `prices` ("
    "  `id` int(11) NOT NULL AUTO_INCREMENT,"
    "  `date` varchar(14) NOT NULL,"
    "  `symbol` varchar(14) NOT NULL,"
    "  `value` varchar(16) NOT NULL,"
    "  PRIMARY KEY (`id`)"
    ") ENGINE=InnoDB")


cnx = mysql.connector.connect(user='root', password='my-secret-pw',
                              host='localhost',
                              port='3307',
                              database=DB_NAME)
cursor = cnx.cursor()

def create_database(cursor):
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)

try:
    cursor.execute("USE {}".format(DB_NAME))
except mysql.connector.Error as err:
    print("Database {} does not exists.".format(DB_NAME))
    if err.errno == errorcode.ER_BAD_DB_ERROR:
        create_database(cursor)
        print("Database {} created successfully.".format(DB_NAME))
        cnx.database = DB_NAME
    else:
        print(err)
        exit(1)


for table_name in TABLES:
    table_description = TABLES[table_name]
    try:
        print("Creating table {}: ".format(table_name), end='')
        cursor.execute(table_description)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("already exists.")
        else:
            print(err.msg)
    else:
        print("OK")

cursor.close()
cnx.close()
