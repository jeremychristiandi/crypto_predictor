import streamlit as st
import pandas as pd
import yfinance as yf
import time

from main_contents import predict_contents

def view_home():
    st.title("Predict Cryptocurrency Price")
    st.caption("Predict cryptocurrency price using a pre-trained model.")
    # selected_ticker = st.text_input("Enter Ticker:", placeholder="Ticker")
    selected_ticker = st.selectbox('Select Ticker', ('BTC', 'USDC', 'XRP'))
    selected_windows = st.slider('Total days of observation', 1, 180, 7)
    selected_horizon = st.slider('Total days of prediction', 1, 30, 1)
    # selected_periods = st.slider("Period (months)", 1, 60, 12)
    # selected_periods = f"{selected_periods}mo"

    if st.button("Generate Prediction"):
        if selected_ticker != "" and selected_ticker != "" and selected_windows != "" and selected_horizon != "":
            # df_temp = yf.download(tickers=selected_ticker,
            #                 period=selected_periods,
            #                 interval="1d")
            # df_temp = df_temp.to_csv("data.csv")
            # inputted_data = pd.read_csv("data.csv")

            df = pd.read_csv(f"./data/{selected_ticker}-USD_5yrs.csv", parse_dates=["Date"], index_col=["Date"])
            predict_contents(df, selected_ticker, selected_windows, selected_horizon)
        else:
            alert = st.warning("Please enter the correct input!", icon="⚠️")
            time.sleep(2)
            alert.empty()

    