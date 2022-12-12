# Get the crypto data:
1. Prepare a [ChromeDriver](https://chromedriver.chromium.org/downloads) in the same folder.

2. Run the command to obtain the crypto data from the [CoinMarketCap](https://coinmarketcap.com/historical/) website:

   ```shell
   python .\CoinData.py --sd 20221110 --ed 20221211 --item 100 --save ./Data/
   ```

# Generate the temp file:

1. Run the command to obtain a temp .pkl file for accelerating back-testing experiments:

   ```shell
   python .\GetTempData.py --sd 20221110 --ed 20221211 --index 10 --save_name Data
   ```

# Crypto data:

This .zip file involves cryptocurrencies data ranging from 2013.04.28 to 2022.11.29 stored in .csv format.
