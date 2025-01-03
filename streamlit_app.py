import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Helper Functions
def calculate_cagr(data, start_date, end_date):
    start_price = data.loc[start_date]
    end_price = data.loc[end_date]
    n_years = (end_date - start_date).days / 365.25
    return ((end_price / start_price) ** (1 / n_years)) - 1

# Sidebar - User Inputs
def sidebar():
    st.sidebar.title("Watchlist")
    watchlist = st.sidebar.text_input("Enter stock tickers (comma-separated):", "AAPL, TSLA")
    return [ticker.strip().upper() for ticker in watchlist.split(',')]

# Main - Interactive Charts
def display_chart(ticker):
    try:
        # Fetching historical stock data
        stock_data = yf.Ticker(ticker).history(period="5y")
        stock_data["50_SMA"] = stock_data["Close"].rolling(window=50).mean()
        stock_data["200_SMA"] = stock_data["Close"].rolling(window=200).mean()
        
        st.title(f"{ticker} - Stock Analysis")
        st.subheader("Historical Price with Moving Averages")

        fig, ax = plt.subplots(figsize=(10, 5))
        stock_data["Close"].plot(label="Close Price", ax=ax)
        stock_data["50_SMA"].plot(label="50-day SMA", ax=ax)
        stock_data["200_SMA"].plot(label="200-day SMA", ax=ax)

        # Linear regression trendline
        reg = LinearRegression()
        stock_data["Days"] = np.arange(len(stock_data))
        reg.fit(stock_data["Days"].values.reshape(-1, 1), stock_data["Close"].fillna(0))
        trendline = reg.predict(stock_data["Days"].values.reshape(-1, 1))
        ax.plot(stock_data.index, trendline, label="Trendline", linestyle="--")

        # Display CAGR
        cagr = calculate_cagr(
            stock_data["Close"].dropna(),
            stock_data.index[0],
            stock_data.index[-1]
        )

        st.pyplot(fig)
        st.write(f"**CAGR:** {cagr:.2%}")

    except Exception as e:
        st.error(f"Unable to fetch data for {ticker}. Error: {e}")

# Main App
st.set_page_config(page_title="Investment Dashboard", layout="wide")
st.title("Interactive Investment Dashboard")

watchlist = sidebar()

# Display charts for each stock in the watchlist
if watchlist:
    for ticker in watchlist:
        display_chart(ticker)
