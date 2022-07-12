from privateKeys.privateData import credentials

credents = credentials()
iexFactors = [
  {
    "symbol":"oilWTI",
    "timeSeriesUrlParam":"/energy/DCOILWTICO",
    'frequency':'W',
    'columnsToKeep':['date', 'value'],
    'columnNames':['date', 'value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"CPI",
    "timeSeriesUrlParam":"/economic/CPIAUCSL",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"FedFunds",
    "timeSeriesUrlParam":"/economic/FEDFUNDS",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"OEF",
    "timeSeriesUrlParam":"HISTORICAL_PRICES/OEF",
    "frequency":"D",
    "columnsToKeep":['date','close'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"SPY",
    "timeSeriesUrlParam":"HISTORICAL_PRICES/SPY",
    "frequency":"D",
    "columnsToKeep":['date','close'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"URTH",
    "timeSeriesUrlParam":"HISTORICAL_PRICES/URTH",
    "frequency":"D",
    "columnsToKeep":['date','close'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },

]