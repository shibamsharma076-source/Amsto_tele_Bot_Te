import streamlit as st
from utils.api_client import get_page_data

st.title("Home")
data = get_page_data("home")

if data["success"]:
    st.write(data["content"])
else:
    st.error(data["error"])
