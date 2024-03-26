import streamlit as st
import pandas as pd
import os
import math

from main_contents import predict_contents

def view_predict_upload():
    st.title(f"Predict Your Own Cryptocurrency Data.")
    uploaded_file = st.file_uploader("Upload your time series .CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    selected_ticker = st.selectbox('Select Ticker', ('BTC', 'USDC', 'XRP'), help="A ticker is a symbol/code representing a token or cryptocurrency.")

    if uploaded_file != None:
        if selected_ticker not in os.path.basename(uploaded_file.name):
            st.write(":red[Are you sure the selected ticker is correct?]")

    total_periods = math.floor(len(df) * 0.1) - 1
    # total_periods = 30
    selected_periods = st.slider("Period (days)", 1, total_periods, 7, help="Total number of days of prediction")

    if selected_ticker == 'BTC':
        st.write(f"Selected ticker: :red[{selected_ticker} (Bitcoin)]")
    elif selected_ticker == 'USDC':
        st.write(f"Selected ticker: :red[{selected_ticker} (USD Coin)]")
    elif selected_ticker == 'XRP':
        st.write(f"Selected ticker: :red[{selected_ticker} (Ripple)]")

    st.write(f"Selected period(s): :red[{selected_periods} days]")

    if uploaded_file is not None:
        if st.button("Generate Prediction"):
            predict_contents(df, ticker=selected_ticker, periods=selected_periods, is_predict=True)