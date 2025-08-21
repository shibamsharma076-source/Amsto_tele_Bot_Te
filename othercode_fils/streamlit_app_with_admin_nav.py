# streamlit_app.py
import streamlit as st
import requests
import json
import re # Import regex for parsing the download info
from streamlit.components.v1 import html # Import html component
import base64
import time
from urllib.parse import urljoin

# --- Page Configuration ---
st.set_page_config(
    page_title="Streamlit + Flask Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --- Chat History Management ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # --- API Configuration ---
# FLASK_API_URL = "http://localhost:5000/chat"
# # Base URL for Flask's media download endpoint
FLASK_MEDIA_DOWNLOAD_BASE_URL = "http://localhost:5000" # Changed to base URL for relative path


















# --- Page Configuration (keep as before) ---
FLASK_API_URL = "http://localhost:5000"  # Base
CHAT_ENDPOINT = urljoin(FLASK_API_URL, "/chat")
LOGIN_ENDPOINT = urljoin(FLASK_API_URL, "/login")
REGISTER_ENDPOINT = urljoin(FLASK_API_URL, "/register")
LOGOUT_ENDPOINT = urljoin(FLASK_API_URL, "/logout")
ADMIN_CREATE_ENDPOINT = urljoin(FLASK_API_URL, "/admin/create")
ADMIN_LIST_ENDPOINT = urljoin(FLASK_API_URL, "/admin/list_users")

# session_state defaults
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# helper for headers
def auth_headers():
    hdr = {}
    


# --- Custom HTML for Speech Recognition ---
# This JavaScript handles the microphone input and sends text back to Streamlit.
# It now positions the microphone button absolutely within the text input's container.
SPEECH_RECOGNITION_HTML = """
<script>
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = SpeechRecognition ? new SpeechRecognition() : null;

    if (recognition) {
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        let finalTranscript = '';
        let isListening = false;
        let pauseTimeout = null;

        // Target the chat input instead of text input
        let textArea = parent.document.querySelector('[data-testid="stChatInput"] textarea');
        let textInputContainer = parent.document.querySelector('[data-testid="stChatInput"]');

        if (textInputContainer && textArea) {
            let micButton = document.getElementById('micButton');
            if (!micButton) {
                micButton = document.createElement('button');
                micButton.id = 'micButton';
                micButton.innerHTML = '<i class="fa-solid fa-microphone"></i>';
                micButton.style.cssText = `
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    font-size: 16px;
                    cursor: pointer;
                    border-radius: 50%;
                    height: 42px;
                    width: 42px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    position: absolute;
                    right: 6px;
                    top: 50%;
                    transform: translateY(-50%);
                    z-index: 10;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                    transition: background-color 0.3s ease;
                `;
                textInputContainer.style.position = 'relative';
                textInputContainer.appendChild(micButton);
                textArea.style.paddingRight = '55px';

                // Click-to-toggle listening
                micButton.addEventListener('click', () => {
                    if (!isListening) {
                        finalTranscript = '';
                        recognition.start();
                        isListening = true;
                        micButton.style.backgroundColor = '#f44336'; // red
                        micButton.innerHTML = '<i class="fa-solid fa-microphone-slash"></i>';
                    } else {
                        stopRecognition();
                    }
                });

                function stopRecognition() {
                    recognition.stop();
                    isListening = false;
                    micButton.style.backgroundColor = '#4CAF50'; // green
                    micButton.innerHTML = '<i class="fa-solid fa-microphone"></i>';
                    if (pauseTimeout) {
                        clearTimeout(pauseTimeout);
                        pauseTimeout = null;
                    }
                }

                recognition.onresult = (event) => {
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        const transcript = event.results[i][0].transcript;
                        if (event.results[i].isFinal) {
                            finalTranscript += transcript;
                        }
                    }
                    if (textArea) {
                        textArea.value = finalTranscript;
                        textArea.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    if (isListening) {
                        micButton.style.backgroundColor = '#f44336';
                        if (pauseTimeout) {
                            clearTimeout(pauseTimeout);
                            pauseTimeout = null;
                        }
                    }
                };

                recognition.onspeechend = () => {
                    if (isListening) {
                        micButton.style.backgroundColor = '#FFC107'; // yellow
                        if (pauseTimeout) clearTimeout(pauseTimeout);
                        pauseTimeout = setTimeout(() => {
                            stopRecognition();
                        }, 5000);
                    }
                };

                recognition.onerror = () => {
                    stopRecognition();
                };

                recognition.onend = () => {
                    if (!isListening) {
                        micButton.style.backgroundColor = '#4CAF50';
                        micButton.innerHTML = '<i class="fa-solid fa-microphone"></i>';
                    }
                };
            }
        }
    } else {
        console.warn('Speech Recognition not supported in this browser.');
    }
</script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
"""




if st.session_state.auth_token:
    st.markdown(f"**Logged in as:** {st.session_state.user_email}")
    st.markdown(f"**Role:** {st.session_state.user_role}")

    # Navigation menu
    menu_options = ["Chat"]
    if st.session_state.user_role == "admin":
        menu_options.append("Admin Dashboard")

    st.session_state.page = st.radio("Navigation", menu_options, key="nav_menu")

    if st.button("Logout", key="logout_button_sidebar"):
        try:
            requests.post(LOGOUT_ENDPOINT, headers=auth_headers(), timeout=10)
        except:
            pass
        st.session_state.auth_token = None
        st.session_state.user_role = None
        st.session_state.user_email = None
        st.rerun()


    else:
        st.info("Please login or register to use the chat")

# If not logged in => show login/register UI
if not st.session_state.auth_token:
    st.header("Sign in or Create an account")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        login_email = st.text_input("Email", key="login_email")
        login_pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login", key="login_btn"):
            try:
                resp = requests.post(LOGIN_ENDPOINT, json={"email": login_email, "password": login_pw}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.auth_token = data["token"]
                    st.session_state.user_role = data["role"]
                    st.session_state.user_email = data["email"]
                    st.rerun()
                else:
                    err = resp.json().get("error") if resp.headers.get("Content-Type", "").startswith("application/json") else resp.text
                    st.error(f"Login failed: {err}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    with tab2:
        reg_email = st.text_input("Email (register)", key="reg_email")
        reg_pw = st.text_input("Password (register)", type="password", key="reg_pw")
        if st.button("Register"):
            try:
                resp = requests.post(REGISTER_ENDPOINT, json={"email": reg_email, "password": reg_pw}, timeout=10)
                if resp.status_code in (200,201):
                    st.success("Registered successfully. Please login.")
                else:
                    err = resp.json().get("error") if resp.headers.get("Content-Type", "").startswith("application/json") else resp.text
                    st.error(f"Registration failed: {err}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.stop()

# Page rendering based on navigation selection
if st.session_state.auth_token:
    if st.session_state.page == "Chat":
        # === Chat UI ===
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_prompt = st.chat_input("Type your message...", key="chat_input_main")
        try:
            html(SPEECH_RECOGNITION_HTML, height=0, width=0)  # Mic button injection
        except:
            pass

    elif st.session_state.page == "Admin Dashboard" and st.session_state.user_role == "admin":
        st.title("Admin Dashboard")

        st.subheader("Create New Admin")
        new_admin_email = st.text_input("New Admin Email")
        new_admin_pw = st.text_input("New Admin Password", type="password")
        if st.button("Create Admin", key="create_admin_btn"):
            resp = requests.post(
                ADMIN_CREATE_ENDPOINT,
                headers=auth_headers(),
                json={"email": new_admin_email, "password": new_admin_pw}
            )
            if resp.status_code == 200:
                st.success("Admin created successfully")
            else:
                st.error(resp.json().get("error", "Failed to create admin"))

        st.subheader("User List")
        resp = requests.get(ADMIN_LIST_ENDPOINT, headers=auth_headers())
        if resp.status_code == 200:
            users = resp.json().get("users", [])
            for u in users:
                st.write(f"📧 {u['email']} | Role: {u['role']}")
        else:
            st.error("Failed to load users")
  # stop rendering the rest until logged in

# At this point user is logged in
# Admin panel link (only visible to admins)
if st.session_state.user_role == "admin":
    st.sidebar.header("Admin Panel")
    if st.sidebar.button("Open Admin Panel"):
        st.session_state.show_admin = True

# Show main chat UI (existing UI) — make sure to keep structure
st.title("Streamlit & Flask Chatbot")
st.markdown("Welcome to the secured chat. Use the chat input below. 🎧")

# display messages as before
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "audio_url" in message:
            st.markdown(message["content"])
            st.audio(message["audio_url"])
            st.download_button(label=f"Download {message['audio_filename']}", data=requests.get(message["audio_url"]).content, file_name=message["audio_filename"], mime="audio/wav")
        else:
            st.markdown(message["content"])




# --- Custom HTML for Speech Recognition ---
# This JavaScript handles the microphone input and sends text back to Streamlit.
# It now positions the microphone button absolutely within the text input's container.
SPEECH_RECOGNITION_HTML = """
<script>
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = SpeechRecognition ? new SpeechRecognition() : null;

    if (recognition) {
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        let finalTranscript = '';
        let isListening = false;
        let pauseTimeout = null;

        // Target the chat input instead of text input
        let textArea = parent.document.querySelector('[data-testid="stChatInput"] textarea');
        let textInputContainer = parent.document.querySelector('[data-testid="stChatInput"]');

        if (textInputContainer && textArea) {
            let micButton = document.getElementById('micButton');
            if (!micButton) {
                micButton = document.createElement('button');
                micButton.id = 'micButton';
                micButton.innerHTML = '<i class="fa-solid fa-microphone"></i>';
                micButton.style.cssText = `
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    font-size: 16px;
                    cursor: pointer;
                    border-radius: 50%;
                    height: 42px;
                    width: 42px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    position: absolute;
                    right: 6px;
                    top: 50%;
                    transform: translateY(-50%);
                    z-index: 10;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                    transition: background-color 0.3s ease;
                `;
                textInputContainer.style.position = 'relative';
                textInputContainer.appendChild(micButton);
                textArea.style.paddingRight = '55px';

                // Click-to-toggle listening
                micButton.addEventListener('click', () => {
                    if (!isListening) {
                        finalTranscript = '';
                        recognition.start();
                        isListening = true;
                        micButton.style.backgroundColor = '#f44336'; // red
                        micButton.innerHTML = '<i class="fa-solid fa-microphone-slash"></i>';
                    } else {
                        stopRecognition();
                    }
                });

                function stopRecognition() {
                    recognition.stop();
                    isListening = false;
                    micButton.style.backgroundColor = '#4CAF50'; // green
                    micButton.innerHTML = '<i class="fa-solid fa-microphone"></i>';
                    if (pauseTimeout) {
                        clearTimeout(pauseTimeout);
                        pauseTimeout = null;
                    }
                }

                recognition.onresult = (event) => {
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        const transcript = event.results[i][0].transcript;
                        if (event.results[i].isFinal) {
                            finalTranscript += transcript;
                        }
                    }
                    if (textArea) {
                        textArea.value = finalTranscript;
                        textArea.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    if (isListening) {
                        micButton.style.backgroundColor = '#f44336';
                        if (pauseTimeout) {
                            clearTimeout(pauseTimeout);
                            pauseTimeout = null;
                        }
                    }
                };

                recognition.onspeechend = () => {
                    if (isListening) {
                        micButton.style.backgroundColor = '#FFC107'; // yellow
                        if (pauseTimeout) clearTimeout(pauseTimeout);
                        pauseTimeout = setTimeout(() => {
                            stopRecognition();
                        }, 5000);
                    }
                };

                recognition.onerror = () => {
                    stopRecognition();
                };

                recognition.onend = () => {
                    if (!isListening) {
                        micButton.style.backgroundColor = '#4CAF50';
                        micButton.innerHTML = '<i class="fa-solid fa-microphone"></i>';
                    }
                };
            }
        }
    } else {
        console.warn('Speech Recognition not supported in this browser.');
    }
</script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
"""



if not st.session_state.auth_token:
    st.header("Sign in or Create an account")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        login_email = st.text_input("Email", key="login_email")
        login_pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login"):
            try:
                resp = requests.post(LOGIN_ENDPOINT, json={"email": login_email, "password": login_pw})
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.auth_token = data["token"]
                    st.session_state.user_role = data["role"]
                    st.session_state.user_email = data["email"]
                    st.rerun()
                else:
                    st.error(resp.json().get("error", "Login failed"))
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        reg_email = st.text_input("Email (register)", key="reg_email")
        reg_pw = st.text_input("Password (register)", type="password", key="reg_pw")
        if st.button("Register"):
            try:
                resp = requests.post(REGISTER_ENDPOINT, json={"email": reg_email, "password": reg_pw})
                if resp.status_code in (200, 201):
                    st.success("Registered successfully. Please log in.")
                else:
                    st.error(resp.json().get("error", "Registration failed"))
            except Exception as e:
                st.error(f"Error: {e}")

    st.stop()

# Page rendering based on navigation selection
if st.session_state.auth_token:
    if st.session_state.page == "Chat":
        # === Chat UI ===
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_prompt = st.chat_input("Type your message...")
        try:
            html(SPEECH_RECOGNITION_HTML, height=0, width=0)  # Mic button injection
        except:
            pass

    elif st.session_state.page == "Admin Dashboard" and st.session_state.user_role == "admin":
        st.title("Admin Dashboard")

        st.subheader("Create New Admin")
        new_admin_email = st.text_input("New Admin Email")
        new_admin_pw = st.text_input("New Admin Password", type="password")
        if st.button("Create Admin", key="create_admin_btn"):
            resp = requests.post(
                ADMIN_CREATE_ENDPOINT,
                headers=auth_headers(),
                json={"email": new_admin_email, "password": new_admin_pw}
            )
            if resp.status_code == 200:
                st.success("Admin created successfully")
            else:
                st.error(resp.json().get("error", "Failed to create admin"))

        st.subheader("User List")
        resp = requests.get(ADMIN_LIST_ENDPOINT, headers=auth_headers())
        if resp.status_code == 200:
            users = resp.json().get("users", [])
            for u in users:
                st.write(f"📧 {u['email']} | Role: {u['role']}")
        else:
            st.error("Failed to load users")


# --- Main UI ---
# st.title("Streamlit & Flask Chatbot")
# st.markdown("This Streamlit app communicates with a Flask backend to get responses. Now with integrated voice input and media download/extraction!")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # If the message has audio_url, display audio player
        if message["role"] == "assistant" and "audio_url" in message:
            st.markdown(message["content"]) # Display text part
            st.audio(message["audio_url"], format="audio/wav") # Display audio player
            st.download_button(
                label=f"Download {message['audio_filename']}",
                data=requests.get(message["audio_url"]).content, # Fetch content again for download button
                file_name=message["audio_filename"],
                mime="audio/wav"
            )
        else:
            st.markdown(message["content"])



# --- User Input and API Call Logic ---
# The chat_input is automatically handled by Streamlit.
# The JavaScript above will populate this input when voice is detected.
# user_prompt = st.chat_input(
#     "Ask ..........."
# )


# Place the custom HTML component at the very end to ensure DOM elements are ready
# The height and width are set to 0 to make the component itself invisible,
# as its purpose is to inject and manipulate other DOM elements.
# Now inject mic button after input exists
# html(SPEECH_RECOGNITION_HTML, height=0, width=0)



if user_prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Display user message in the chat container
    with st.chat_message("user"):
        st.markdown(user_prompt)
    

    # Make the API call to the Flask backend
    try:
        with st.spinner("Processing your request..."):
            # response = requests.post(
            #     FLASK_API_URL,
            #     data={'message': user_prompt},
            # )
            response = requests.post(CHAT_ENDPOINT, data={"message": user_prompt}, headers=auth_headers())
        
        if response.status_code == 200:
            ai_response_data = response.json()
            ai_response_text = ai_response_data.get('response', 'No response from API.')
            download_info = ai_response_data.get('download_info') # Get download info if present
            
            if download_info and download_info.get('type') == 'audio_tts':
                # Handle TTS audio response specifically
                audio_url = f"{FLASK_MEDIA_DOWNLOAD_BASE_URL}{download_info['download_url']}"
                audio_filename = download_info['filename']
                
                # Add AI response text and audio info to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response_text,
                    "audio_url": audio_url,
                    "audio_filename": audio_filename
                })
                with st.chat_message("assistant"):
                    st.markdown(ai_response_text)
                    st.audio(audio_url, format="audio/wav")
                    st.download_button(
                        label=f"Download {audio_filename}",
                        data=requests.get(audio_url).content, # Fetch content again for download button
                        file_name=audio_filename,
                        mime="audio/wav"
                    )
            elif download_info and download_info.get('type') == 'media_download':
                # Handle general media downloads
                download_relative_url = download_info['download_url']
                download_filename = download_info['filename']
                
                full_download_url = f"{FLASK_MEDIA_DOWNLOAD_BASE_URL}{download_relative_url}"
                
                # Inject JavaScript to programmatically click a hidden link
                js_code = f"""
                <script>
                    var link = document.createElement('a');
                    link.href = "{full_download_url}";
                    link.download = "{download_filename}"; // Suggests filename to browser
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    console.log('Attempted programmatic download for {download_filename}');
                </script>
                """
                html(js_code, height=0, width=0) # Inject the script (invisible component)
                
                # Add a fallback message to the chat indicating download initiation
                fallback_message = (
                    f"If the download for '{download_filename}' doesn't start automatically, "
                    f"you can [click here to download]({full_download_url})."
                )
                st.session_state.messages.append({"role": "assistant", "content": ai_response_text + "\n\n" + fallback_message})
                with st.chat_message("assistant"):
                    st.markdown(ai_response_text + "\n\n" + fallback_message)
            else:
                # Handle QR/Barcode image responses
                qr_image_b64 = ai_response_data.get('qr_image_base64')
                barcode_image_b64 = ai_response_data.get('barcode_image_base64')
                uploaded_image_b64 = ai_response_data.get('uploaded_image_base64')
                download_info = ai_response_data.get('download_info')

                

                st.session_state.messages.append({"role": "assistant", "content": ai_response_text})


                # If no specific download info, just show the AI response text
                recipe_image_url = ai_response_data.get('recipe_image_url')
                recipe_video_url = ai_response_data.get('recipe_video_url')
                
                with st.chat_message("assistant"):
                    st.markdown(ai_response_text)
                    # Show recipe image if present
                    if recipe_image_url:
                        if recipe_image_url.startswith("/"):
                            recipe_image_url = FLASK_MEDIA_DOWNLOAD_BASE_URL + recipe_image_url
                        st.image(recipe_image_url, caption="Recipe Image", use_container_width=True)
                        st.markdown(f"[🔗 Open Recipe Image]({recipe_image_url})", unsafe_allow_html=True)
                        st.write("DEBUG image_url:", recipe_image_url)
                    # Show clickable video link if present
                    if recipe_video_url:
                        st.markdown(f"▶️ [Watch Video]({recipe_video_url})", unsafe_allow_html=True)
                    # Show QR code image if present
                    if qr_image_b64:
                        st.image(base64.b64decode(qr_image_b64), caption="QR Code", use_container_width=False)
                    # Show Barcode image if present
                    if barcode_image_b64:
                        st.image(base64.b64decode(barcode_image_b64), caption="Barcode", use_container_width=False)
                    # Show uploaded image if decoding
                    if uploaded_image_b64:
                        st.image(base64.b64decode(uploaded_image_b64), caption="Uploaded QR/Barcode", use_container_width=False)
                    # Show download link if present
                    if download_info and download_info.get('download_url'):
                        full_download_url = f"{FLASK_MEDIA_DOWNLOAD_BASE_URL}{download_info['download_url']}"
                        download_filename = download_info.get('filename', 'download.png')
                        st.markdown(
                            f"[⬇️ Download {download_filename}]({full_download_url})",
                            unsafe_allow_html=True
                        )

        else:
            error_message = response.json().get('error', 'Unknown error from API.')
            st.error(f"Error from Flask API: {error_message}")


    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Is the Flask server running at {FLASK_API_URL}?")
        st.info("Please run `python flask_app.py` in a separate terminal before running this app.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


if st.session_state.user_role == "admin":
    st.header("Admin Panel")
    new_admin_email = st.text_input("New Admin Email")
    new_admin_pw = st.text_input("New Admin Password", type="password")
    if st.button("Create Admin"):
        resp = requests.post(ADMIN_CREATE_ENDPOINT, headers=auth_headers(), json={"email": new_admin_email, "password": new_admin_pw})
        if resp.status_code == 200:
            st.success("Admin created successfully")
        else:
            st.error(resp.json().get("error", "Failed to create admin"))



