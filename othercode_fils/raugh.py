













# Sidebar navigation for all users
page_selection = st.sidebar.radio("Navigate", ["Home", "About Us", "Documentation", "Contact", "FAQ", "Chat"])

if page_selection != "Chat":
    data = get_page_data(page_selection.lower().replace(" ", ""))
    st.title(page_selection)
    if data["success"]:
        st.write(data["content"])
    else:
        st.error(data["error"])
    st.stop()  # Stop further rendering for non-chat pages

































@app.route('/test_plagiarism', methods=['POST'])
def test_plagiarism():
    """
    A dedicated endpoint to test local and API-based plagiarism detection.
    """
    if not PLAGIARISM_CORPUS_LOADED:
        return jsonify({"error": "Local plagiarism corpus not loaded. Please ensure 'plagiarism_corpus.csv' exists."}), 500

    try:
        data = request.get_json()
        text_to_check = data.get('text', '').strip()

        if not text_to_check:
            return jsonify({"error": "No text provided for plagiarism check."}), 400

        # Local plagiarism check
        plagiarized_segments = local_plagiarism_check(text_to_check, plagiarism_corpus_df)
        if plagiarized_segments:
            highlighted = text_to_check
            for seg in plagiarized_segments:
                highlighted = highlighted.replace(seg["matched_text"], f"**:red[{seg['matched_text']}]**")
            return jsonify({
                "status": "plagiarism_detected",
                "input_text": text_to_check,
                "highlighted_text": highlighted,
                "similarity_scores": [round(seg['similarity'], 2) for seg in plagiarized_segments]
            })
        else:
            # API fallback (LLM-based)
            api_result = api_plagiarism_check(text_to_check)
            if api_result and "plagiarism" in api_result:
                return jsonify({
                    "status": "api_checked",
                    "input_text": text_to_check,
                    "api_result": api_result['plagiarism']
                })
            else:
                return jsonify({
                    "status": "no_plagiarism",
                    "input_text": text_to_check,
                    "message": "No plagiarism detected in your text (local corpus and API)."
                })

    except Exception as e:
        return jsonify({"error": f"An error occurred during plagiarism testing: {str(e)}"}), 500



























































uploaded_files = st.file_uploader(
            "📁", # A simple emoji icon for the button
            type=["txt", "pdf", "docx", "csv", "json", "png", "jpg"],
            accept_multiple_files=True,
            # label_visibility="collapsed"
        )










# --- API Keys and URLs ---
# IMPORTANT: Replace these with your actual API keys.
# For production, consider using environment variables (e.g., os.environ.get("GEMINI_API_KEY"))
# and never commit them directly to version control.
API_KEYS = {
    "gemini": "AIzaSyBSMFk5zk0HkZnia5QpVdRFvmd4FBpUvqY", # Your Gemini API Key
    "openai": "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", # Replace with your OpenAI API Key
    "groq": "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",     # Replace with your Groq API Key
    "microsoft_phi3": "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", # Replace with your Microsoft Phi-3 API Key
    "tencent_hunyuan": "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", # Replace with your Tencent API Key
    "copilot": "c50657ae2b294851b1f2cdf39355a444.ef9de966c39ec31c" # Replace with your Copilot API Key (if direct API available)
}

API_ENDPOINTS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent",
    "openai": "https://api.openai.com/v1/chat/completions", # Using chat completions endpoint
    "groq": "https://api.groq.com/openai/v1/chat/completions", # Groq uses OpenAI-compatible API
    "microsoft_phi3": "https://api.microsoft.com/v1/models/phi-3-medium-128k-instruct/generate", # Hypothetical/Example
    "tencent_hunyuan": "https://api.tencent.com/v1/models/hunyuan-a13b-instruct/generate", # Hypothetical/Example
    "copilot": "https://api.copilot.com/v1/generate" # Hypothetical/Example
}

# Define the preferred order of LLM APIs to try
LLM_PRIORITY_ORDER = ["gemini", "openai", "groq", "microsoft_phi3", "tencent_hunyuan", "copilot"]








# --- Define all keyword lists at the beginning of the function ---
        # These lists MUST be defined here to be in scope for all subsequent conditional checks.
         code_keywords = ["correct", "correct the code", "analyse the code", "analyze code", "detect code language", "fix this code", "fix it", "code language", "code correction", "code analysis", "debug", "error in code"]
        
        disease_keywords = ["detect disease", "disease", "predict disease", "symptoms", "disease prediction", "disease detection", "disease symptoms", "predict disease symptoms", "disease analysis", "disease diagnose", "disease diagnosis", "disease prediction symptoms", "disease diagnose symptoms", "disease diagnose analysis", "disease diagnosis analysis", "disease diagnose prediction", "disease diagnosis prediction"]
        
        study_keywords = ["study", "qualification", "how to study", "guid", "guide", "process", "university", "apply job", "job process", "job qualification", "job study",]
        
        sentiment_keywords = ["sentiment", "analyse sentiment", "analyze sentiment", "sentiment analysis", "sentiment detect", "sentiment detection", "sentiment score", "sentiment score analysis", "sentiment score detect", "sentiment score detection"]
        
        fake_news_keywords = ["fake", "news", "is fake", "is news", "fack check", "real or fake", "verify news", "verify article", "verify news article", "verify news content", "verify article content"]
        
        job_recommendation_keywords = ["jobe", "preferred job", "job recommendation", "job suggestions", "job advice", "career recommendation", "career advice", "job search", "find job", "job opportunities"]

        spam_keywords = ["spam", "junk", "unsolicited", "advertisement", "buy now", "click here", "subscribe", "free offer", "winner", "claim prize", "limited time", "urgent", "act now", "risk-free", "guaranteed", "no cost", "exclusive deal", "special promotion", "unsecured credit", "debt relief", "make money fast", "work from home", "earn cash", "get paid to click", "online business opportunity"] 

        # Keywords for media handling
        media_task_keywords = ["download", "download video", "video download", "audio download", "download thumbnail", "extract video script", "extract video tags", "extract video views and earnings", "extract video description", "extract video title", "extract video content","extract video metadata", "extract audio", "extract video", "extract video content", "extract video metadata", "extract video transcript", "extract video subtitles" "extract video captions", "extract video comments", "extract video likes","extract video shares", "extract video statistics", "extract video information", "extract video details", "extract video data", "extract video insights", "extract video summary", "extract video highlights", "extract video key points"]
        
        # Keywords for Interview Q&A
        interview_keywords = ["interview", "interview q&a", "q&a", "job q&a", "job interview", "interview questions", "interview questions and answers", "job interview questions", "job interview q&a", "job interview questions and answers", "interview questions for", "interview questions for job", "job interview questions for", "interview questions for job role", "interview questions for job role", "interview questions for job position", "job interview questions for job position"]

        # Keywords for Text Translation
        translation_keywords = ["translate", "translate this" "translate the text", "translate the article",
                               "translate the content", "translate the document", "translate the input", "translate the input text",
                               "translate the input article", "translate the input document", "translate the input content",
                               "text to translate", "translate this text", "translate this article", "translate this content", "translate this document"]

        # Keywords for Text Summarization
        summarization_keywords = ["summarise", "summaries the article", "article summarise", "summarize", "summarize this", "summarize the text", "summarize the article", "summarize the content", "summarize the document", "summarize the text", "summarize the input", "summarize the input text", "summarize the input article", "summarize the input document", "summarize the input content"]

        # Keywords for Text-to-Speech
        tts_keywords = ["audio", "text to audio", "speech", "text to speech", "text to speech", "text to speak", "speak this", "read this out loud"]
        voice_role_keywords = ["male", "female", "child", "matured man", "alpha male", "alpha female", "mature female", "boy", "younger boy", "girl", "younger girl", "soft man", "soft women", "rude men", "rude women", "default"]
        #Plagiarism detection keywords
        plagiarism_keywords = ["plagiarism", "check plagiarism", "plagiarism check", "detect plagiarism", "plagiarism detection", "is this plagiarised", "is this original", "originality check", "is this content original", "is this content plagiarised", "plagiarism free", "plagiarism checker", "detect plagiarism", "plagiarism", "plagiarism detect", "detect is an ai writed", "detect is non human writed", "detect is an human writed"]

        # --- DEBUGGING AID: Confirm keyword lists are defined ---
























elif "jobe" in user_message.lower() or "obe" in user_message.lower() or "ob" in user_message.lower() or "jop" in user_message.lower() or "job" in user_message.lower() or "preferred job" in user_message.lower() or "prefer jobe" in user_message.lower() or "preferred jobe" in user_message.lower():




























# === HEADER STYLES & FUNCTION ===
# HEADER_STYLE = """
# <style>
# .header-bar {
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     background-color: #1E1E1E;
#     padding: 10px 20px;
#     color: white;
#     font-family: Arial, sans-serif;
#     position: fixed;
#     top: 0;
#     width: 100%;
#     z-index: 999;
# }
# .header-left {
#     position: absolute;
#     left: 20px;
#     display: flex;
#     align-items: center;
# }
# .header-left img {
#     height: 35px;
#     margin-right: 10px;
# }
# .header-center {
#     display: flex;
#     gap: 20px;
# }
# .header-center a {
#     color: white;
#     text-decoration: none;
#     font-weight: bold;
# }
# .header-center a:hover {
#     text-decoration: underline;
# }
# .profile-container {
#     position: absolute;
#     right: 20px;
# }
# .profile-icon {
#     height: 35px;
#     width: 35px;
#     border-radius: 50%;
#     cursor: pointer;
# }
# .dropdown-menu {
#     display: none;
#     position: absolute;
#     right: 0;
#     top: 40px;
#     background-color: #fff;
#     color: black;
#     min-width: 180px;
#     box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
#     z-index: 1000;
#     border-radius: 5px;
# }
# .dropdown-menu a {
#     display: block;
#     padding: 10px;
#     color: black;
#     text-decoration: none;
# }
# .dropdown-menu a:hover {
#     background-color: #f2f2f2;
# }
# .profile-container:hover .dropdown-menu {
#     display: block;
# }
# </style>
# """

# def render_header():
    # st.markdown(HEADER_STYLE, unsafe_allow_html=True)

    # if not st.session_state.get("auth_token"):  # Not logged in
    #     st.markdown(f"""
    #     <div class="header-bar">
    #         <div class="header-left">
    #             <img src="https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg" alt="Logo">
    #             <span style="font-size:18px; font-weight:bold;">My Web App</span>
    #         </div>
    #         <div class="header-center">
    #             <a href="#">Home</a>
    #             <a href="#">About Us</a>
    #             <a href="#">Documentation</a>
    #             <a href="#">Contact Us</a>
    #             <a href="#">FAQ's</a>
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    # else:  # Logged in
    #     st.markdown(f"""
    #     <div class="header-bar">
    #         <div class="header-left">
    #             <img src="https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg" alt="Logo">
    #             <span style="font-size:18px; font-weight:bold;">My Web App</span>
    #         </div>
    #         <div class="profile-container">
    #             <img src="https://www.w3schools.com/howto/img_avatar.png" class="profile-icon" alt="Profile">
    #             <div class="dropdown-menu">
    #                 <a href="#">Profile Page</a>
    #                 <a href="#">Settings</a>
    #                 <a href="#">About Us</a>
    #                 <a href="#">Contact Us</a>
    #                 <a href="#">FAQ's</a>
    #             </div>
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)


# render_header()
# st.markdown("<br><br><br>", unsafe_allow_html=True)  # add spacer so main content not hidden






























# streamlit_app.py

import streamlit as st
import requests
import json
from io import BytesIO

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
FLASK_API_URL = "http://localhost:5000/chat"

# --- Main UI ---
st.title("Streamlit & Flask Chatbot")
st.markdown("This Streamlit app communicates with a Flask backend to get responses.")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input, File Upload, and API Call Logic ---
# The form ensures all inputs (text and files) are submitted at once.
with st.form(key="chat_form", clear_on_submit=True):
    # Use columns to create a visually integrated input area
    col1, col2, col3 = st.columns([1, 15, 1])

    with col1:
        # The file uploader is our "button" to open the file dialog.
        # We set label_visibility="collapsed" to hide the label.
        uploaded_files = st.file_uploader(
            "📁", # A simple emoji icon for the button
            type=["txt", "pdf", "docx", "csv", "json", "png", "jpg"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        # You can add a tooltip for a better user experience
        st.write("Click to upload files")

    with col2:
        # The text area for the user's message
        user_prompt = st.text_area(
            "type input text here",
            placeholder="Type your message here...",
            label_visibility="collapsed"
        )

    with col3:
        # The submit button for the form
        submitted = st.form_submit_button("➤", use_container_width=True)

if submitted and (user_prompt or uploaded_files):
    # Add user message to chat history
    user_display = user_prompt
    if uploaded_files:
        filenames = ", ".join([f.name for f in uploaded_files])
        user_display += f"\n\nFiles attached: {filenames}"
    
    st.session_state.messages.append({"role": "user", "content": user_display})
    
    # Display user message in the chat container
    with st.chat_message("user"):
        st.markdown(user_display)

    # Make the API call to the Flask backend
    try:
        with st.spinner("Waiting for Flask backend..."):
            # Construct the payload
            payload = {'message': user_prompt}
            
            # Prepare the files for the request
            files_to_send = [
                ('files', (file.name, BytesIO(file.getvalue()), file.type))
                for file in uploaded_files
            ]

            response = requests.post(
                FLASK_API_URL,
                data=payload,
                files=files_to_send
            )

        # Check if the request was successful
        if response.status_code == 200:
            ai_response = response.json().get('response', 'No response from API.')
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)
        else:
            st.error(f"Error from Flask API: Status Code {response.status_code}")
            st.error(response.text)

    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Is the Flask server running at {FLASK_API_URL}?")
        st.info("Please run `python flask_app.py` in a separate terminal before running this app.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")









































# streamlit_app.py

import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Gemini Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- API Configuration ---
FLASK_API_URL = "http://localhost:5000/chat"

# --- Main UI ---
st.title("Streamlit & Flask Chatbot with Gemini")
st.markdown("This app uses a Flask backend to access the Gemini API for intelligent responses.")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and API Call Logic ---
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])

    with col1:
        user_prompt = st.text_input(
            "type input text here",
            placeholder="Type your message here...",
            label_visibility="collapsed"
        )
    
    with col2:
        submitted = st.form_submit_button("➤", use_container_width=True)

if submitted and user_prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Display user message in the chat container
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Make the API call to the Flask backend without any files
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





















































# flask_app.py
import os
import json
import requests
import time
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# The API key is automatically provided in the canvas environment.
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
API_KEY = "AIzaSyBSMFk5zk0HkZnia5QpVdRFvmd4FBpUvqY" # The canvas environment will inject the API key here.

# Paths to our local model files
VECTORIZER_PATH = 'vectorizer.pkl'
CLASSIFIER_PATH = 'classifier.pkl'

# --- Load the local model on startup ---
try:
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(CLASSIFIER_PATH, 'rb') as f:
        classifier = pickle.load(f)
    print("Local spam classifier loaded successfully.")
    LOCAL_MODEL_LOADED = True
except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"WARNING: Could not load local model files. Reason: {e}")
    print("Spam classification will fall back to Gemini API.")
    LOCAL_MODEL_LOADED = False

def call_gemini_api(prompt):
    """
    Makes a call to the Gemini API with exponential backoff.
    
    Args:
        prompt (str): The text prompt for the model.
    
    Returns:
        str: The generated text response from the API.
    """
    if not API_KEY:
        raise ValueError("API key is missing. Please provide a valid key.")

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    
    headers = {'Content-Type': 'application/json'}
    
    retries = 3
    delay = 1
    for i in range(retries):
        try:
            response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", json=payload, headers=headers)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "I'm sorry, I couldn't generate a response from the API."
        except requests.exceptions.HTTPError as e:
            if i < retries - 1 and (e.response.status_code == 429 or e.response.status_code >= 500):
                print(f"API call failed with status code {e.response.status_code}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e
        except Exception as e:
            raise e

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat requests, prioritizing local spam classification.
    """
    try:
        user_message = request.form.get('message', '').strip()
        
        # --- Priority 1: Spam Classification ---
        if any(keyword in user_message.lower() for keyword in ["spam", "ist spam", "fraud", "not spam"]):
            local_response = "I couldn't classify your message locally."
            
            if LOCAL_MODEL_LOADED:
                try:
                    message_transformed = vectorizer.transform([user_message])
                    prediction = classifier.predict(message_transformed)[0]
                    
                    if prediction:
                        local_response = f"Local model says: This message is likely '{prediction}'."
                        return jsonify({'response': local_response})
                except Exception as e:
                    print(f"Error during local model prediction: {e}")
            
            # If local model failed or was not loaded, fall back to Gemini
            try:
                prompt = f"Classify the following text as 'Spam' or 'Not Spam' and provide a brief reason:\n\n{user_message}"
                gemini_response = call_gemini_api(prompt)
                return jsonify({'response': gemini_response})
            except Exception as e:
                return jsonify({'response': f"Both local and API services failed to classify the message. Error: {str(e)}"}), 500
        
        # --- Priority 2: Job Recommendations ---
        elif "jobe" in user_message.lower() or "obe" in user_message.lower() or "ob" in user_message.lower() or "jop" in user_message.lower() or "job" in user_message.lower() or "preferred job" in user_message.lower() or "prefer jobe" in user_message.lower() or "preferred jobe" in user_message.lower():
            prompt = f"""
            Based on the following user-provided text, analyze the skills and qualifications.
            Suggest a list of 3 suitable job roles. For each role, provide:
            1. A brief reason why it's a good fit.
            2. 3 specific companies to consider applying to.
            3. The best time of year to apply for these types of roles.
            
            User Text: "{user_message}"
            """
            ai_response = call_gemini_api(prompt)
            return jsonify({'response': ai_response})
        
        # --- Priority 3: General Chat ---
        else:
            if user_message:
                prompt = f"Respond to the following user message: {user_message}"
                ai_response = call_gemini_api(prompt)
                return jsonify({'response': ai_response})
            else:
                return jsonify({'response': "I'm sorry, I didn't receive a message."})
    
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"API Error: {e}"}), 500
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/test_spam_model', methods=['POST'])
def test_spam_model():
    """
    A dedicated endpoint to test the local spam classification model.
    """
    if not LOCAL_MODEL_LOADED:
        return jsonify({"error": "Local model not loaded. Please ensure 'train_model.py' has been run."}), 500
    
    try:
        data = request.get_json()
        text_to_check = data.get('text', '')
        
        if not text_to_check:
            return jsonify({"error": "No text provided for classification."}), 400
        
        # Transform the input text using the loaded vectorizer
        message_transformed = vectorizer.transform([text_to_check])
        
        # Get the prediction from the classifier
        prediction = classifier.predict(message_transformed)[0]
        
        # Return the prediction in a JSON response
        return jsonify({
            "status": "success",
            "text": text_to_check,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during local model testing: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)





































# streamlit_app.py

import streamlit as st
import requests
import json
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Gemini Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- API Configuration ---
FLASK_API_URL = "http://localhost:5000/chat"

# --- Main UI ---
st.title("Streamlit & Flask Chatbot with Gemini")
st.markdown("This app uses a Flask backend to access the Gemini API for intelligent responses, including context-aware recommendations.")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input, File Upload, and API Call Logic ---
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])

    with col1:
        user_prompt = st.text_input(
            "type input text here",
            placeholder="Type your message here...",
            label_visibility="collapsed"
        )
    
    with col2:
        submitted = st.form_submit_button("➤", use_container_width=True)

if submitted and user_prompt:
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
                data={
                    'message': user_prompt,
                    # 'location' parameter is removed here
                },
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

