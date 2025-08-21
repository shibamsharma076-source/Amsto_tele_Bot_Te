# streamlit_app.py
import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Streamlit + Flask Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- API Configuration ---
# The URL of our Flask backend API.
# It's important to use the correct port. Flask runs on 5000 by default.
FLASK_API_URL = "http://localhost:5000/chat"

# --- Main UI ---
st.title("Streamlit & Flask Chatbot")
st.markdown("This Streamlit app communicates with a Flask backend to get responses.")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and API Call Logic ---
if user_prompt := st.chat_input("Ask the Flask API a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Display user message in the chat container
    with st.chat_message("user"):
        st.markdown(user_prompt)
    # Make the API call to the Flask backend
    try:
        with st.spinner("Waiting for Gemini response..."):
            response = requests.post(
                FLASK_API_URL,
                data={'message': user_prompt},
            )
        
        if response.status_code == 200:
            ai_response = response.json().get('response', 'No response from API.')
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)
        else:
            error_message = response.json().get('error', 'Unknown error from API.')
            st.error(f"Error from Flask API: {error_message}")


    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Is the Flask server running at {FLASK_API_URL}?")
        st.info("Please run `python flask_app.py` in a separate terminal before running this app.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
