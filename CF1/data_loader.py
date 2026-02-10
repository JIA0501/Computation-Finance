import pandas as pd
from pandas_datareader import data as pdr


def download_prices_stooq(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    candidates = [
        ticker,
        ticker.upper(),
        f"{ticker.upper()}.US",
        f"{ticker.upper()}.us",
    ]

    last_err = None
    for t in candidates:
        try:
            df = pdr.DataReader(t, "stooq", start, end)
            if df is not None and not df.empty:
                df = df.sort_index()  # oldest -> newest
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to download data for {ticker} from Stooq. "
        f"Tried: {candidates}. Last error: {last_err}"
    )