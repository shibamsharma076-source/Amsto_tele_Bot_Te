# streamlit_app.py
import os
import streamlit as st
import requests
import json
import re # Import regex for parsing the download info
from streamlit.components.v1 import html # Import html component
import base64
import time
from urllib.parse import urljoin

from pathlib import Path



from extra_streamlit_components import CookieManager

cookie_manager = CookieManager()


# # --- API Configuration ---
# FLASK_API_URL = "http://localhost:5000/chat"
# # Base URL for Flask's media download endpoint
FLASK_MEDIA_DOWNLOAD_BASE_URL = "http://localhost:5000" # Changed to base URL for relative path








# --- Page Configuration (keep as before) ---
# FLASK_API_URL = "http://localhost:5000"  # Base


FLASK_API_URL = os.environ.get("FLASK_API_URL")
if not FLASK_API_URL:
    # Fallback for local development
    # FLASK_API_URL = "http://localhost:5000"
    FLASK_API_URL = "http://51.20.128.67:5000"  # EC2 instance IP

CHAT_ENDPOINT = urljoin(FLASK_API_URL, "/chat")
LOGIN_ENDPOINT = urljoin(FLASK_API_URL, "/login")
REGISTER_ENDPOINT = urljoin(FLASK_API_URL, "/register")
LOGOUT_ENDPOINT = urljoin(FLASK_API_URL, "/logout")
ADMIN_CREATE_ENDPOINT = urljoin(FLASK_API_URL, "/admin/create")
ADMIN_LIST_ENDPOINT = urljoin(FLASK_API_URL, "/admin/list_users")















# API helper
def get_page_data(page_name):
    try:
        resp = requests.get(f"{FLASK_API_URL}/page-content/{page_name}", headers=auth_headers(), timeout=5)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"success": False, "error": f"Error {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


























def auth_headers():
    token = st.session_state.get("auth_token")
    if token:
        return {"X-Auth-Token": token}
    return {}





# ------------------------
# HEADER RENDERING FUNCTION
# ------------------------
# def render_header():
#     """
#     Top header that adapts to login status.
#     """
#     svg_logo = """
#     <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
#       <rect width="24" height="24" rx="5" fill="#2563EB"></rect>
#       <path d="M6 12h12" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
#       <path d="M6 8h12" stroke="white" stroke-width="1.8" stroke-linecap="round" opacity="0.6"/>
#       <path d="M6 16h8" stroke="white" stroke-width="1.8" stroke-linecap="round" opacity="0.6"/>
#     </svg>
#     """


#     # Always show welcome message, no profile icon for logged-in users
#     header_right_html = """<span style="color:#6b7280;font-weight:600;font-size:14px;">Welcome</span>"""



#     # Full HTML + CSS for header
#     header_html = f"""
#     <style>
#     .top-header {{
#         width: 100%;
#         background: #fff;
#         border-bottom: 1px solid #ddd;
#         padding: 10px 18px;
#         display: flex;
#         align-items: center;
#         justify-content: space-between;
#         position: sticky;
#         top: 0;
#         z-index: 9999;
#     }}
#     .header-left {{ display: flex; align-items: center; gap: 10px; }}
#     .app-name {{ font-weight: 700; font-size: 18px; color: #111827; }}
#     .header-center {{
#         position: absolute;
#         left: 50%;
#         transform: translateX(-50%);
#         display: flex;
#         gap: 22px;
#         align-items: center;
#     }}
#     .header-center a {{
#         color: #374151;
#         text-decoration: none;
#         font-weight: 600;
#         font-size: 14px;
#     }}
#     .header-center a:hover {{
#         color: #1e40af;
#     }}
#     .header-right {{ display: flex; align-items: center; gap: 12px; }}
#     .profile-icon {{
#         width: 40px;
#         height: 40px;
#         border-radius: 50%;
#         background: linear-gradient(180deg,#111827,#374151);
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         color: white;
#         font-weight: 700;
#         cursor: pointer;
#         position: relative;
#     }}
#     .profile-dropdown {{
#         position: absolute;
#         right: 0;
#         top: 48px;
#         background: white;
#         border: 1px solid #ddd;
#         border-radius: 8px;
#         display: none;
#         flex-direction: column;
#         width: 180px;
#     }}
#     .profile-dropdown a {{
#         padding: 8px 12px;
#         text-decoration: none;
#         color: #111827;
#         font-weight: 600;
#     }}
#     .profile-dropdown a:hover {{ background: #f3f4f6; }}
#     .profile-wrapper:hover .profile-dropdown {{ display: flex; }}
#     </style>

#     <div class="top-header">
#         <div class="header-left">
#             <div class="logo">{svg_logo}</div>
#             <div class="app-name">YourAppName</div>
#         </div>
#         <div class="header-center">
#             <a href="#home">Home</a>
#             <a href="#about">About Us</a>
#             <a href="#docs">Documentation</a>
#             <a href="#contact">Contact</a>
#             <a href="#faq">FAQ</a>
#         </div>
#         <div class="header-right">
#             {header_right_html}
#         </div>
#     </div>
#     """
#     st.markdown(header_html, unsafe_allow_html=True)

# ------------------------
# End header rendering
# ------------------------

# def render_header():
    # st.markdown(
    #     """
    #     <style>
    #     .header {
    #         display: flex;
    #         justify-content: center;
    #         gap: 20px;
    #         background-color: white;
    #         padding: 10px;
    #         border-bottom: 1px solid #ddd;
    #     }
    #     .header a {
    #         text-decoration: none;
    #         color: #333;
    #         font-weight: bold;
    #     }
    #     .header a:hover {
    #         color: #007BFF;
    #     }
    #     </style>
    #     <div class="header">
    #         <a href="?page=Home">Home</a>
    #         <a href="?page=About Us">About Us</a>
    #         <a href="?page=Documentation">Documentation</a>
    #         <a href="?page=Contact Us">Contact Us</a>
    #         <a href="?page=FAQ">FAQ</a>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )





def render_header():
    left, center, right = st.columns([1, 4, 1])

    with left:
        st.markdown("### Amsto")  # App name
        # Or load logo:
        # logo_path = Path(__file__).parent / "static" / "logo.png"
        # st.image(str(logo_path), width=40)

    # with center:
    #     nav_items = ["Home", "About Us", "Documentation", "Contact", "FAQ"]
    #     nav_urls = ["#home", "#about", "#docs", "#contact", "#faq"]
    #     cols = st.columns(len(nav_items))
    #     for i, (name, url) in enumerate(zip(nav_items, nav_urls)):
    #         with cols[i]:
    #             if st.button(name, key=f"nav_{name}"):
    #                 st.session_state["current_page"] = name  # store page
    #                 st.experimental_rerun()

    # with right:
        # if st.session_state.get("auth_token"):
        #     st.write(f"Hello, {st.session_state.get('user_email', '')}")
        # else:
        #     st.write("Welcome")

        with right:
            if st.session_state.get("auth_token"):
                # Profile icon with dropdown
                st.markdown(
                    f"""
                    <style>
                    .profile-wrapper {{
                        position: relative;
                        display: inline-block;
                    }}
                    .profile-icon {{
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        background: linear-gradient(180deg,#111827,#374151);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-weight: bold;
                        cursor: pointer;
                    }}
                    .profile-dropdown {{
                        display: none;
                        position: absolute;
                        right: 0;
                        top: 48px;
                        background: white;
                        border: 1px solid #ddd;
                        border-radius: 6px;
                        min-width: 180px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                        z-index: 1000;
                    }}
                    .profile-dropdown p, .profile-dropdown button {{
                        margin: 0;
                        padding: 10px;
                        border: none;
                        background: none;
                        width: 100%;
                        text-align: left;
                        font-size: 14px;
                        cursor: pointer;
                    }}
                    .profile-dropdown p {{
                        font-weight: bold;
                        color: #111827;
                    }}
                    .profile-dropdown button:hover {{
                        background: #f3f4f6;
                    }}
                    .profile-wrapper:hover .profile-dropdown {{
                        display: block;
                    }}
                    </style>
                    <div class="profile-wrapper">
                        <div class="profile-icon">{st.session_state.get('user_email', '')[:1].upper()}</div>
                        <div class="profile-dropdown">
                            <p>Logged in as: {st.session_state.get('user_email', '')}</p>
                            <p>Role: {st.session_state.get('user_role', '')}</p>
                            <form action="" method="post">
                                <button type="submit" name="logout">Logout</button>
                            </form>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Handle logout click
                if st.button("Logout", key="header_logout"):
                    try:
                        requests.post(LOGOUT_ENDPOINT, headers=auth_headers(), timeout=3)
                    except:
                        pass
                    cookie_manager.delete("auth_token")
                    cookie_manager.delete("user_email")
                    cookie_manager.delete("user_role")
                    st.session_state.clear()
                    st.rerun()
            else:
                st.write("Welcome")


















# --- Page Configuration ---
st.set_page_config(
    page_title="Amsto",
    layout="wide",
    initial_sidebar_state="collapsed"
)




# --- Chat History Management ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []







# # Call header rendering right after page config
if not st.session_state.get("auth_token"):
    render_header()











# session_state defaults
if "auth_token" not in st.session_state:
    st.session_state.auth_token = cookie_manager.get("auth_token")
    st.session_state.user_email = cookie_manager.get("user_email")
    st.session_state.user_role = cookie_manager.get("user_role")
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "messages" not in st.session_state:
    st.session_state.messages = []










# Always sync cookies into session_state
cookie_auth_token = cookie_manager.get("auth_token")
cookie_user_email = cookie_manager.get("user_email")
cookie_user_role = cookie_manager.get("user_role")

if cookie_auth_token:
    st.session_state.auth_token = cookie_auth_token
    st.session_state.user_email = cookie_user_email
    st.session_state.user_role = cookie_user_role











# Top bar: show user info / logout if logged in
with st.sidebar:
    # st.title("Account")
    if st.session_state.auth_token:
        st.markdown(f"**Logged in as:** {st.session_state.user_email}")
        # st.markdown(f"**Role:** {st.session_state.user_role}")
        # if st.button("Logout", key="logout_btn"):
        #     try:
        #         requests.post(LOGOUT_ENDPOINT, headers=auth_headers(), timeout=10)
        #     except:
        #         pass
        #     st.session_state.auth_token = None
        #     st.session_state.user_role = None
        #     st.session_state.user_email = None
        #     st.rerun()
        if st.button("Logout", key="logout_btn"):
            try:
                requests.post(LOGOUT_ENDPOINT, headers=auth_headers(), timeout=3)
            except:
                pass
            cookie_manager.delete("auth_token")
            cookie_manager.delete("user_email")
            cookie_manager.delete("user_role")
            st.session_state.clear()
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

                    # Set cookies
                    cookie_manager.set("auth_token", data["token"])
                    cookie_manager.set("user_email", data["email"])
                    cookie_manager.set("user_role", data["role"])


                    st.rerun()
                else:
                    err = resp.json().get("error") if resp.headers.get("Content-Type", "").startswith("application/json") else resp.text
                    st.error(f"Login failed: {err}")
            except Exception as e:
                st.error(f"Connection error: {e}")



    with tab2:
        reg_full_name = st.text_input("Full Name", key="reg_full_name")
        reg_dob = st.date_input("Date of Birth", key="reg_dob")
        reg_username = st.text_input("Username", key="reg_username")
        reg_email = st.text_input("Email (register)", key="reg_email")
        reg_pw = st.text_input("Password (register)", type="password", key="reg_pw")

        if st.button("Register"):
            try:
                payload = {
                    "full_name": reg_full_name,
                    "dob": str(reg_dob),
                    "username": reg_username,
                    "email": reg_email,
                    "password": reg_pw
                }
                resp = requests.post(REGISTER_ENDPOINT, json=payload, timeout=10)
                if resp.status_code in (200, 201):
                    st.success("Registered successfully. Please login.")
                else:
                    err = resp.json().get("error") if resp.headers.get("Content-Type", "").startswith("application/json") else resp.text
                    st.error(f"Registration failed: {err}")
            except Exception as e:
                st.error(f"Connection error: {e}")




    st.stop()  # stop rendering the rest until logged in

# At this point user is logged in
# Admin panel link (only visible to admins)
if st.session_state.user_role == "admin":
    st.sidebar.header("Admin Panel")
    if st.sidebar.button("Open Admin Panel"):
        st.session_state.show_admin = True





# not requir
# Show main chat UI (existing UI) — make sure to keep structure
# st.title("Streamlit & Flask Chatbot")
# st.markdown("Welcome to the secured chat. Use the chat input below. 🎧")

# # display messages as before
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         if message["role"] == "assistant" and "audio_url" in message:
#             st.markdown(message["content"])
#             st.audio(message["audio_url"])
#             st.download_button(label=f"Download {message['audio_filename']}", data=requests.get(message["audio_url"]).content, file_name=message["audio_filename"], mime="audio/wav")
#         else:
#             st.markdown(message["content"])





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



# === Sidebar Navigation for Admins ===
if st.session_state.get("user_role") == "admin":
    page = st.sidebar.radio("Navigation", ["Chat", "Admin Dashboard"], key="nav_radio")
else:
    page = "Chat"






# --- Main UI ---
# === Render Pages (Chat and Admin Dashboard) ===

if page == "Chat":
    # *** Chat UI (keeps your original structure) ***
    st.title("Welcome..")
    st.markdown("Welcome to the secured chat. Use the chat input below. 🎧")

    # display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "audio_url" in message:
                st.markdown(message["content"])
                st.audio(message["audio_url"])
                # Download button uses unique key to avoid duplication issues
                st.download_button(
                    label=f"Download {message['audio_filename']}",
                    data=requests.get(message["audio_url"]).content,
                    file_name=message["audio_filename"],
                    mime="audio/wav",
                    key=f"dl_{message.get('audio_filename','no_name')}"
                )
            else:
                st.markdown(message["content"])

    # Inject mic button HTML (do this only on Chat page)
    # try:
    #     html(SPEECH_RECOGNITION_HTML, height=0, width=0)
    # except Exception:
    #     pass

    # *** Chat input – add a unique key to avoid DuplicateElementId ***
    user_prompt = st.chat_input("Ask ...........", key="chat_input_main")

    # --- User prompt handling (same logic as before) ---
    if user_prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Display user message in the chat container
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Make the API call to the Flask backend
        try:
            with st.spinner("Processing your request..."):
                response = requests.post(CHAT_ENDPOINT, data={"message": user_prompt}, headers=auth_headers())

            if response.status_code == 200:
                ai_response_data = response.json()
                ai_response_text = ai_response_data.get('response', 'No response from API.')
                download_info = ai_response_data.get('download_info')  # Get download info if present

                # Handle TTS audio response
                if download_info and download_info.get('type') == 'audio_tts':
                    audio_url = f"{FLASK_MEDIA_DOWNLOAD_BASE_URL}{download_info['download_url']}"
                    audio_filename = download_info['filename']
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
                            data=requests.get(audio_url).content,
                            file_name=audio_filename,
                            mime="audio/wav",
                            key=f"dl_{audio_filename}"
                        )

                # Handle general media downloads (programmatic click)
                elif download_info and download_info.get('type') == 'media_download':
                    download_relative_url = download_info['download_url']
                    download_filename = download_info['filename']
                    full_download_url = f"{FLASK_MEDIA_DOWNLOAD_BASE_URL}{download_relative_url}"

                    js_code = f"""
                    <script>
                        var link = document.createElement('a');
                        link.href = "{full_download_url}";
                        link.download = "{download_filename}";
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    </script>
                    """
                    html(js_code, height=0, width=0)

                    fallback_message = (
                        f"If the download for '{download_filename}' doesn't start automatically, "
                        f"you can [click here to download]({full_download_url})."
                    )
                    st.session_state.messages.append({"role": "assistant", "content": ai_response_text + "\n\n" + fallback_message})
                    with st.chat_message("assistant"):
                        st.markdown(ai_response_text + "\n\n" + fallback_message)

                else:
                    # Generic response + images (QR/Barcode/Recipe)
                    qr_image_b64 = ai_response_data.get('qr_image_base64')
                    barcode_image_b64 = ai_response_data.get('barcode_image_base64')
                    uploaded_image_b64 = ai_response_data.get('uploaded_image_base64')
                    recipe_image_url = ai_response_data.get('recipe_image_url')
                    recipe_video_url = ai_response_data.get('recipe_video_url')

                    st.session_state.messages.append({"role": "assistant", "content": ai_response_text})

                    with st.chat_message("assistant"):
                        st.markdown(ai_response_text)

                        if recipe_image_url:
                            if recipe_image_url.startswith("/"):
                                recipe_image_url = FLASK_MEDIA_DOWNLOAD_BASE_URL + recipe_image_url
                            st.image(recipe_image_url, caption="Recipe Image", use_container_width=True)
                            st.markdown(f"[🔗 Open Recipe Image]({recipe_image_url})", unsafe_allow_html=True)

                        if recipe_video_url:
                            st.markdown(f"▶️ [Watch Video]({recipe_video_url})", unsafe_allow_html=True)

                        if qr_image_b64:
                            st.image(base64.b64decode(qr_image_b64), caption="QR Code", use_container_width=False)

                        if barcode_image_b64:
                            st.image(base64.b64decode(barcode_image_b64), caption="Barcode", use_container_width=False)

                        if uploaded_image_b64:
                            st.image(base64.b64decode(uploaded_image_b64), caption="Uploaded QR/Barcode", use_container_width=False)

                        if download_info and download_info.get('download_url'):
                            full_download_url = f"{FLASK_MEDIA_DOWNLOAD_BASE_URL}{download_info['download_url']}"
                            download_filename = download_info.get('filename', 'download.png')
                            st.markdown(f"[⬇️ Download {download_filename}]({full_download_url})", unsafe_allow_html=True)

            else:
                # non-200 from chat endpoint
                try:
                    error_message = response.json().get('error', 'Unknown error from API.')
                except Exception:
                    error_message = response.text
                st.error(f"Error from Flask API: {error_message}")

        except requests.exceptions.ConnectionError:
            st.error(f"Connection Error: Is the Flask server running at {FLASK_API_URL}?")
            st.info("Please run `python flask_app.py` in a separate terminal before running this app.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")




# --- Admin Dashboard (separate page) ---
elif page == "Admin Dashboard" and st.session_state.get("user_role") == "admin":
    st.header("Admin Dashboard")
    st.write("Create new admins or manage users (admin-only).")

    # Create Admin
    new_admin_email = st.text_input("New Admin Email", key="admin_new_email")
    new_admin_pw = st.text_input("New Admin Password", type="password", key="admin_new_pw")
    if st.button("Create Admin", key="create_admin_btn"):
        resp = requests.post(ADMIN_CREATE_ENDPOINT, headers=auth_headers(),
                             json={"email": new_admin_email, "password": new_admin_pw})
        if resp.status_code == 200:
            st.success("Admin created successfully")
        else:
            try:
                st.error(resp.json().get("error", "Failed to create admin"))
            except Exception:
                st.error("Failed to create admin (unexpected response)")

    st.markdown("---")

    # List Users
    if st.button("List Users", key="list_users_btn"):
        resp = requests.get(ADMIN_LIST_ENDPOINT, headers=auth_headers())
        if resp.status_code == 200:
            users = resp.json().get("users", [])
            if users:
                st.table(users)
            else:
                st.info("No users found")
        else:
            try:
                st.error(resp.json().get("error", "Failed to fetch users"))
            except Exception:
                st.error("Failed to fetch users (unexpected response)")


