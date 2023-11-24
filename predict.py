import streamlit as st
import yfinance as yf
import pandas as pd
import time

from main_contents import predict_contents

def view_predict():
    st.title("Predict Cryptocurrency Price")
    st.caption("Get **real-time** cryptocurrency prediction!")
    selected_ticker = st.text_input("Enter Ticker:", placeholder="[Ticker]-USD")
    selected_periods = st.slider("Period (months)", 1, 60, 12)
    selected_periods = f"{selected_periods}mo"

    if st.button("Generate Prediction"):
        if selected_ticker != "":
            df_temp = yf.download(tickers=selected_ticker,
                            period=selected_periods,
                            interval="1d")
            df_temp = df_temp.to_csv("data.csv")
            df = pd.read_csv("data.csv")
            predict_contents(df)
        else:
            alert = st.warning("Something is wrong with the input!", icon="⚠️")
            time.sleep(2)
            alert.empty()