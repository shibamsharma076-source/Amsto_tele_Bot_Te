import requests
import streamlit as st

FLASK_API_URL = "http://127.0.0.1:5000"

def get_page_data(page_name):
    try:
        resp = requests.get(f"{FLASK_API_URL}/page-content/{page_name}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"success": False, "error": f"Error {resp.status_code}: {resp.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}
