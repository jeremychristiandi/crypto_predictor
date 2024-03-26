import streamlit as st
from streamlit_option_menu import option_menu

from home import view_home
from predict import view_predict_upload
from faq import faq_contents
from articles import view_articles
from about import view_about

selected = option_menu(
    menu_title="Cryptocurrency Predictor",
    options=["Home", "Predict", "FAQ", "News", "About"],
    icons=["house-door", "journal", "newspaper", "info-circle"], # bootstrap icon,
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

def handleSelectedPage(selected):
    if selected == "Home":
        view_home()
    if selected == "Predict":
        view_predict_upload()
    if selected == "FAQ":
        faq_contents()
    if selected == "News":
        view_articles()
    if selected == "About":
        view_about()

handleSelectedPage(selected)