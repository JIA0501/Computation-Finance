#Excise 1 step 1,2,3
import datetime as dt
from data_loader import download_prices_stooq
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset
from skfolio.preprocessing import prices_to_returns

ticker = "MSFT"
start_date = "2006-01-01"
end_date = dt.date.today().strftime("%Y-%m-%d")

data = download_prices_stooq(ticker, start_date, end_date)
stock = data["Close"].astype(float)

#Price series from market data
open_ = data["Open"].astype(float)
high =  data['High'].astype(float)
low = data['Low'].astype(float)
close = data['Close'].astype(float)

# log returns
r = np.log(close).diff().dropna()


mu_est_d = r.mean() # Daily mean
rv_est_d = ((r-mu_est_d) ** 2).mean() # Realized variance
sigma_est_d = np.sqrt(rv_est_d) # Daily volatility


mu_est_a = mu_est_d * 252 #Annualized daily mean
sigma_est_a = sigma_est_d * np.sqrt(252) # Annualized daily volatility

print(f"[{ticker}] mu_estimated (Annual) = {mu_est_a:.2%}")
print(f"[{ticker}] sigma_estimated (Annual) = {sigma_est_a:.2%}")

#Altenative volatility estimatoers: Parkinson and Garman-Klass
parkinson_var = (1.0/(4.0*np.log(2.0))) * (np.log(high/low) ** 2)
garman_klass_var = 0.5 * (np.log(high / low) ** 2) - (2.0 * np.log(2.0) - 1.0) * (np.log(close / open_) ** 2)

#30 days rolling window
window = 30

rolling_rv = r.rolling(window).var(ddof = 0)

rolling_vol = np.sqrt(rolling_rv) * np.sqrt(252)
rolling_parkinson = np.sqrt(parkinson_var.rolling(window).mean()) * np.sqrt(252)
rolling_gk = np.sqrt(garman_klass_var.rolling(window).mean()) * np.sqrt(252)

# 6) Plot
plt.figure()
rolling_vol.plot(label="Realized (Close-based)")
rolling_parkinson.plot(label="Parkinson (H/L)")
rolling_gk.plot(label="Garman-Klass (OHLC)")
plt.title(f"{ticker} {window}D Rolling Annualized Volatility")
plt.legend()
plt.tight_layout()
plt.show()

#Step 3
# Load SP500 20 assets prices/returns and 3M ATM implied vol
prices = load_sp500_dataset()                 # DataFrame: dates x tickers
implied_vol = load_sp500_implied_vol_dataset()  # DataFrame: dates x tickers (implied vol)

print("prices columns:", list(prices.columns))
print("implied vol columns:", list(implied_vol.columns))

#Pick Microsoft stock form implied volatility dataset
col = "MSFT"
if col not in implied_vol.columns:
    col = implied_vol.columns[0]
    print("AAPL not found. Using:", col)

iv = implied_vol[col].dropna()

rv = rolling_vol.dropna()

common_start = max(rv.index.min(), iv.index.min())
common_end   = min(rv.index.max(), iv.index.max())

rv_aligned = rv.loc[common_start:common_end]
iv_aligned = iv.loc[common_start:common_end]

# Plot
plt.figure()
rv_aligned.plot(label="30D rolling realized vol (ann.)")
iv_aligned.plot(label="3M ATM implied vol")
plt.title(f"{col}: Realized vs Implied Volatility")
plt.legend()
plt.tight_layout()
plt.show()