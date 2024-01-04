import streamlit as st
import pandas as pd

from main_contents import predict_contents

def view_predict_upload():
    st.title(f"Predict Your Own Cryptocurrency Data.")
    uploaded_file = st.file_uploader("Upload your time series .CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        predict_contents(df)