import streamlit as st
from utils.api_client import get_page_data

st.title("Contact")
data = get_page_data("contact")

if data["success"]:
    st.write(data["content"])
else:
    st.error(data["error"])
