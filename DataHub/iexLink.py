from privateKeys.privateData import credentials
from dataLink import dataLink

"""
Grab last item in stock table in order to get the date
get list of business days using pandas
at each business day, append to stock table, and then append stock table to larger stock table.
then upload this to sql
"""

link = dataLink(credentials().credentials)


