import streamlit as st
import pandas as pd
import yfinance as yf
import time

from main_contents import predict_contents

def view_home():
    st.title("Predict Cryptocurrency Price")
    st.caption("Predict cryptocurrency price using a pre-trained model.")
    selected_ticker = st.selectbox('Select Ticker', ('BTC', 'USDC', 'XRP'))
    # selected_windows = st.slider('Total days of observation', 1, 180, 7)
    # selected_horizon = st.slider('Total days of prediction', 1, 30, 1)

    df = pd.read_csv(f"./data/{selected_ticker}-USD_5yrs.csv", parse_dates=["Date"], index_col=["Date"])
    split_size = int(len(df) * 0.9)
    date = df.index.values[split_size:][1:]
    date = pd.to_datetime(date)

    selected_periods = st.slider("Period (days)", 1, 180, 7)

    st.write(f"Selected ticker: :red[{selected_ticker}]")
    st.write(f"Selected period(s): :red[{date[0].strftime('%d/%m/%Y')} to {date[selected_periods-1].strftime('%d/%m/%Y')}]")

    if st.button("Generate Prediction"):
        if selected_ticker != "" and selected_periods != "":
            # df = pd.read_csv(f"./data/{selected_ticker}-USD_5yrs.csv", parse_dates=["Date"], index_col=["Date"])
            predict_contents(df, selected_ticker, selected_periods)
        else:
            alert = st.warning("Please enter the correct input!", icon="⚠️")
            time.sleep(2)
            alert.empty()

    