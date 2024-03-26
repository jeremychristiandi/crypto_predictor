import streamlit as st
import pandas as pd
import time
from datetime import timedelta, datetime

from main_contents import predict_contents

def view_home():
    st.title("Predict Cryptocurrency Price")
    st.caption("Predict cryptocurrency price using a pre-trained deep learning model.")
    st.caption("*For research purposes only, not a financial advice.*")
    selected_ticker = st.selectbox('Select Ticker', ('BTC', 'USDC', 'XRP'), help="A ticker is a symbol/code representing a token or cryptocurrency.")
    # selected_windows = st.slider('Total days of observation', 1, 180, 7)
    # selected_horizon = st.slider('Total days of prediction', 1, 30, 1)

    df = pd.read_csv(f"./data/{selected_ticker}-USD_5yrs.csv", parse_dates=["Date"], index_col=["Date"])
    split_size = int(len(df) * 0.9)
    date = df.index.values[split_size:][1:]
    date = pd.to_datetime(date)

    selected_periods = st.slider("Period (days)", 1, 180, 7, help="Total number of days of prediction")

    if selected_ticker == 'BTC':
        st.write(f"Selected ticker: :red[{selected_ticker} (Bitcoin)]")
    elif selected_ticker == 'USDC':
        st.write(f"Selected ticker: :red[{selected_ticker} (USD Coin)]")
    elif selected_ticker == 'XRP':
        st.write(f"Selected ticker: :red[{selected_ticker} (Ripple)]")

    first_date = date[0].strftime('%d/%m/%Y') 
    first_date_str = datetime.strptime(first_date, '%d/%m/%Y')
    last_date = first_date_str + timedelta(days=selected_periods-1) 
    last_date = last_date.strftime('%d/%m/%Y') 

    # st.write(f"Selected period(s): :red[{date[0].strftime('%d/%m/%Y')} to {date[selected_periods-1].strftime('%d/%m/%Y')}]")
    st.write(f"Selected period(s): :red[{first_date} to {last_date}]")

    if st.button("Generate Prediction"):
        if selected_ticker != "" and selected_periods != "":
            # df = pd.read_csv(f"./data/{selected_ticker}-USD_5yrs.csv", parse_dates=["Date"], index_col=["Date"])
            predict_contents(df, selected_ticker, selected_periods)
        else:
            alert = st.warning("Please enter the correct input!", icon="⚠️")
            time.sleep(2)
            alert.empty()

    