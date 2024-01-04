import streamlit as st

EMAIL = "jeremy.christiandi@binus.ac.id"

def view_about():
    st.title("About")
    st.write("Hello! This application is created by **Jeremy Christiandi**, a student from BINUS University.")
    st.write("Contact me for further questions:")
    st.write("Email: ", EMAIL)