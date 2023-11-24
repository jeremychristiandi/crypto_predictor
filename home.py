import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import yfinance as yf

from main_contents import predict_contents

def view_home():
    st.title(f"Predict Your Time Series Data.")
    uploaded_file = st.file_uploader("Upload your time series .CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        predict_contents(df)