#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import os

def download_data(tickers, start_date, end_date, interval="1wk"):
    """
    Downloads historical stock data (Adjusted Close) from Yahoo Finance.
    """
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval, threads=True)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            prices = data.iloc[:, 0]
    else:
        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            prices = data
    return prices

def preprocess_data(prices):
    """
    Preprocesses price data: drop missing values and sort by date.
    """
    prices_clean = prices.dropna().sort_index()
    return prices_clean

if __name__ == "__main__":
    tickers = ["MSFT", "MMM", "HSY", "GE", "GOOGL", "AMZN", "SHY"]
    start_date = "2012-01-01"
    end_date = "2023-12-31"
    
    prices = download_data(tickers, start_date, end_date, interval="1wk")
    prices_clean = preprocess_data(prices)
    
    # Ensure directory exists
    os.makedirs("data/processed", exist_ok=True)
    prices_clean.to_csv("data/processed/historical_prices_cleaned.csv")
    print("Cleaned data saved to data/processed/historical_prices_cleaned.csv")
