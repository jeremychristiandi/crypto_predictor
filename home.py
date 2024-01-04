import streamlit as st
import pandas as pd
import yfinance as yf
import time

from main_contents import predict_contents

def view_home():
    st.title("Predict Cryptocurrency Price")
    st.caption("Predict cryptocurrency price using a pre-trained model.")
    selected_ticker = st.text_input("Enter Ticker:", placeholder="Ticker") + '-USD'
    selected_periods = st.slider("Period (months)", 1, 60, 12)
    selected_periods = f"{selected_periods}mo"

    if st.button("Generate Prediction"):
        if selected_ticker != "":
            df_temp = yf.download(tickers=selected_ticker,
                            period=selected_periods,
                            interval="1d")
            df_temp = df_temp.to_csv("data.csv")
            inputted_data = pd.read_csv("data.csv")
            predict_contents(inputted_data)
        else:
            alert = st.warning("Something is wrong with the input!", icon="⚠️")
            time.sleep(2)
            alert.empty()

    