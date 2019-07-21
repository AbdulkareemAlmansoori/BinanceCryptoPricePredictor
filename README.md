# Abdulkareem Almansoori

## Binance Cryptocurrency Price Predictor
Predict any cryptocurrency with any pair (USDT/ETH/BTC/BNB/ETC) on Binance through Deep Learning (commonly known as Artificial Intelligence for those who do not know the difference between AI/Ml/Dl). 

Uses python-binance to connect to the Binance account to fetch data. 

### Before starting:
Create an API key via [Binance API](https://www.binance.com/userCenter/createApi.html) and replace the strings with your API & secret key on the script at line 13. 
````bash
# Client credentials (must be changed) -> Binance through python-binance (line 13 @ check.py)
client = Client('API_KEY', 'SECRET_KEY')

# Start
python check.py <pair>

# Example:
python check.py XRPUSDT
python check.py BTCUSDT
python check.py XRPUSD
````