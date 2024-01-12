import streamlit as st
import requests

API_KEY = "pub_35053be17cf98ac299aa66f8f9f7270e4f16d"
URL = f"https://newsdata.io/api/1/news?apikey={API_KEY}&q=crypto"

def view_articles():
    with st.spinner("Loading latest news..."):
        response = requests.get(URL, verify=False)
    status = response.status_code

    if status == 200:
        st.title("Latest News")
        data = response.json()["results"]
        for i in range(0, 10):
            published_date = ", ".join(data[i].get("pubDate").split(" "))

            container = st.container()
            container.title(data[i].get("title"))
            container.caption(published_date)
            if(data[i].get("image_url") != None):
                container.image(data[i].get("image_url"))
            container.write(data[i].get("description"))
            container.write("*Read full article*: " + data[i].get("link"))
    else:
        st.title("Sorry! There are no news for the time being.")
