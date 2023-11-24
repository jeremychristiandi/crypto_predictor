import streamlit as st
from streamlit_option_menu import option_menu

from home import view_home
from predict import view_predict
from faq import faq_contents

selected = option_menu(
    menu_title="Cryptocurrency Predictor",
    options=["Home", "Predict", "FAQ", "About"],
    icons=["house-door", "binoculars", "journal", "info-circle"], # bootstrap icon,
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

if selected == "Home":
    view_predict()
if selected == "Predict":
    view_home()
if selected == "FAQ":
    faq_contents()
if selected == "About":
    st.title(f"You are in {selected} menu")