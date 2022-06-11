from privateKeys.privateData import credentials

credents = credentials()
iexFactors = [
  {
    "symbol":"oilWTI",
    "timeSeriesUrlParam":"/energy/DCOILWTICO",
    'frequency':'W',
    'columnsToKeep':['date', 'value'],
    'columnNames':['date', 'oilWTI'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"CPI",
    "timeSeriesUrlParam":"/economic/CPIAUCSL",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','CPI'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"FedFunds",
    "timeSeriesUrlParam":"/economic/FEDFUNDS",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','FedFunds'],
    'tableName':credents.mainFactorTable
  },

]