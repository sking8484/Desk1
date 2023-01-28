from privateKeys.privateData import credentials

credents = credentials()
iexFactors = [
  {
    "symbol":"GDP",
    "timeSeriesUrlParam":"/economic/A191RL1Q225SBEA",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"oilWTI",
    "timeSeriesUrlParam":"/energy/DCOILWTICO",
    'frequency':'W',
    'columnsToKeep':['date', 'value'],
    'columnNames':['date', 'value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"GAS",
    "timeSeriesUrlParam":"/energy/GASREGCOVW",
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
  {
    "symbol":"3MTreas",
    "timeSeriesUrlParam":"treasury/DGS3MO",
    "frequency":"D",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"10YTreas",
    "timeSeriesUrlParam":"treasury/DGS10",
    "frequency":"D",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"IndProIndex",
    "timeSeriesUrlParam":"/economic/INDPRO",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"recessionIndicator",
    "timeSeriesUrlParam":"/economic/RECPROUSM156N",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"InstitutionalMoneyFunds",
    "timeSeriesUrlParam":"/economic/WIMFSL",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"InitialClaims",
    "timeSeriesUrlParam":"/economic/IC4WSA",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"totalHousingStarts",
    "timeSeriesUrlParam":"/economic/HOUST",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"totalPayrolls",
    "timeSeriesUrlParam":"/economic/PAYEMS",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"totalVehicleSales",
    "timeSeriesUrlParam":"/economic/TOTALSA",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"unemploymentRate",
    "timeSeriesUrlParam":"/economic/UNRATE",
    "frequency":"M",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
  {
    "symbol":"mortgageRates",
    "timeSeriesUrlParam":"/mortgage/MORTGAGE15US",
    "frequency":"W",
    "columnsToKeep":['date','value'],
    "columnNames":['date','value'],
    'tableName':credents.mainFactorTable
  },
]