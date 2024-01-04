import streamlit as st
import requests

from datetime import datetime, date

API_KEY = "pub_35053be17cf98ac299aa66f8f9f7270e4f16d"
URL = "https://newsdata.io/api/1/news?apikey=pub_35053be17cf98ac299aa66f8f9f7270e4f16d&q=crypto"

def view_articles():
    response = requests.get(URL)
    data = response.json()["results"]

    st.title("Latest News")
    if(data != None):
        for i in range(0, 10):
            published_date = ", ".join(data[i].get("pubDate").split(" "))

            container = st.container(border=True)
            container.title(data[i].get("title"))
            container.caption(published_date)
            if(data[i].get("image_url") != None):
                container.image(data[i].get("image_url"))
            container.write(data[i].get("description"))
            container.write("**Read full article**: " + data[i].get("link"))
        