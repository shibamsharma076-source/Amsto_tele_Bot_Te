# flask_app.py
import os
import json
import requests
import time
import pickle
import pandas as pd
from flask import Flask, request, jsonify, send_file, after_this_request
from flask_cors import CORS
import re
import yt_dlp # Import yt-dlp library
import tempfile # For creating temporary files
import shutil # For removing directories
from dotenv import load_dotenv # Import load_dotenv
from bs4 import BeautifulSoup # Import BeautifulSoup for web scraping
import io # For handling in-memory binary data (audio)
import wave # For WAV file creation
import numpy as np # For audio data manipulation
import base64 # For decoding audio data
import difflib
import qrcode
from PIL import Image
import io
import barcode
from barcode.writer import ImageWriter
from pyzbar.pyzbar import decode as pyzbar_decode
import threading

import sqlite3
import uuid
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from bcrypt import hashpw, gensalt, checkpw
from requests_oauthlib import OAuth2Session
from flask_pymongo import PyMongo
import urllib3

from werkzeug.security import generate_password_hash, check_password_hash



urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import nltk
nltk.download('punkt')



# New imports for local TTS and Summarization
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    print("WARNING: 'pyttsx3' library not found. Local Text-to-Speech will be unavailable.")
    PYTTSX3_AVAILABLE = False
    try:
        os.system("pip install pyttsx3")
        import pyttsx3
        PYTTSX3_AVAILABLE = True
        print("Successfully installed 'pyttsx3'.")
    except Exception as e:
        print(f"Could not install 'pyttsx3' automatically: {e}")


# Attempt to import NLTK for local text summarization
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
    # Download NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    print("NLTK 'punkt' and 'stopwords' data checked/downloaded.")
except ImportError:
    print("WARNING: 'nltk' library not found. Local Text Summarization will be unavailable.")
    NLTK_AVAILABLE = False
    try:
        os.system("pip install nltk")
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import sent_tokenize, word_tokenize
        NLTK_AVAILABLE = True
        nltk.download('punkt')
        nltk.download('stopwords')
        print("Successfully installed 'nltk' and downloaded 'punkt', 'stopwords'.")
    except Exception as e:
        print(f"Could not install 'nltk' automatically: {e}")


# Load environment variables from .env file
load_dotenv()

# Attempt to import langdetect for local language detection.
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0 # Seed for reproducibility
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("WARNING: 'langdetect' library not found. Language detection will rely solely on LLM APIs, which may be slower or less precise for this specific task.")
    LANGDETECT_AVAILABLE = False
    try:
        # Attempt to install langdetect if not found
        os.system("pip install langdetect")
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        LANGDETECT_AVAILABLE = True
        print("Successfully installed 'langdetect'.")
    except Exception as e:
        print(f"Could not install 'langdetect' automatically: {e}")


app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing








# --- API Keys and Endpoints ---
# IMPORTANT: Now loading API keys from environment variables
API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY", ""),
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "groq": os.getenv("GROQ_API_KEY", ""),
    "microsoft_phi3": os.getenv("MICROSOFT_PHI3_API_KEY", ""), # Hypothetical
    "tencent_hunyuan": os.getenv("TENCENT_HUNYUAN_API_KEY", ""), # Hypothetical
    "copilot": os.getenv("COPILOT_API_KEY", "") # Hypothetical
}

API_ENDPOINTS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent",
    "gemini_tts": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent", # Specific endpoint for Gemini TTS
    "openai": "https://api.openai.com/v1/chat/completions", # Using OpenAI's chat completions endpoint
    "groq": "https://api.groq.com/openai/v1/chat/completions", # Groq uses an OpenAI-compatible API
    "microsoft_phi3": "https://api.microsoft.com/v1/models/phi-3-medium-128k-instruct/generate", # Hypothetical endpoint
    "tencent_hunyuan": "https://api.tencent.com/v1/models/hunyuan-a13b-instruct/generate", # Hypothetical endpoint
    "copilot": "https://api.copilot.com/v1/generate" # Hypothetical endpoint
}

# Define the preferred order of LLM APIs to try for orchestration
LLM_PRIORITY_ORDER = ["gemini", "openai", "groq", "microsoft_phi3", "tencent_hunyuan", "copilot"]





# Use your Mongo Atlas connection string
app.config["MONGO_URI"] = os.getenv("MONGO_URI")

# "mongodb+srv://<username>:<password>@<clustername>.mongodb.net/<dbname>?retryWrites=true&w=majority"
mongo = PyMongo(app)

MONGO_URI = os.getenv("MONGO_URI") # Get from environment variable
client = MongoClient(MONGO_URI)
db = client["ai_bot"] # Choose a name for your new database









# --- Paths to our local model files ---
SPAM_VECTORIZER_PATH = 'train_Modules/pkl_files/vectorizer.pkl'
SPAM_CLASSIFIER_PATH = 'train_Modules/pkl_files/classifier.pkl'
DISEASE_DATASET_PATH = 'train_Modules/csv_datasetes/disease_dataset.csv'
DISEASE_CLASSIFIER_PATH = 'train_Modules/pkl_files/disease_classifier.pkl'
SYMPTOM_COLUMNS_PATH = 'train_Modules/pkl_files/symptom_columns.pkl'
STUDY_GUIDE_DATASET_PATH = 'train_Modules/csv_datasetes/study_guide_dataset.csv'
SENTIMENT_VECTORIZER_PATH = 'train_Modules/pkl_files/sentiment_vectorizer.pkl'
SENTIMENT_CLASSIFIER_PATH = 'train_Modules/pkl_files/sentiment_classifier.pkl' # Corrected path if different from spam
CODE_ANALYSIS_DATASET_PATH = 'train_Modules/csv_datasetes/code_analysis_dataset.csv'
FAKE_NEWS_DATASET_PATH = 'train_Modules/csv_datasetes/fake_news_dataset.csv'
FAKE_NEWS_VECTORIZER_PATH = 'train_Modules/pkl_files/fake_news_vectorizer.pkl'
FAKE_NEWS_CLASSIFIER_PATH = 'train_Modules/pkl_files/fake_news_classifier.pkl'
INTERVIEW_QA_DATASET_PATH = 'train_Modules/csv_datasetes/interview_qa_dataset.csv' # New: Path to interview Q&A dataset
PLAGIARISM_CORPUS_PATH = 'train_Modules/csv_datasetes/plagiarism_corpus.csv' # Path to the local plagiarism corpus dataset
# Code Generation Dataset Path
CODE_GENERATION_DATASET_PATH = 'train_Modules/csv_datasetes/code_generation_dataset.csv' # New: Path to code generation dataset
# Recipe Dataset Path
RECIPE_DATASET_PATH = 'train_Modules/csv_datasetes/recipes_dataset.csv'


# --- Limits ---
TRANSLATION_WORD_LIMIT = 1800 # Set the word limit for translation
SUMMARIZATION_INPUT_WORD_LIMIT = 1800 # Set the word limit for text input to summarization
TTS_INPUT_WORD_LIMIT = 1800 # Set the word limit for text input to TTS

# --- Load the local models on startup ---
SPAM_MODEL_LOADED = False
try:
    with open(SPAM_VECTORIZER_PATH, 'rb') as f:
        spam_vectorizer = pickle.load(f)
    with open(SPAM_CLASSIFIER_PATH, 'rb') as f:
        spam_classifier = pickle.load(f)
    print("Local spam classifier loaded successfully.")
    SPAM_MODEL_LOADED = True
except (FileNotFoundError, pickle.UnpackingError) as e:
    print(f"WARNING: Could not load local spam model files. Reason: {e}. Spam classification will fall back to LLM APIs.")

# Attempt to load the local disease classifier and dataset
DISEASE_MODEL_LOADED = False
try:
    with open(DISEASE_CLASSIFIER_PATH, 'rb') as f:
        disease_classifier = pickle.load(f)
    with open(SYMPTOM_COLUMNS_PATH, 'rb') as f:
        symptom_columns_list = pickle.load(f)
    
    try:
        disease_df = pd.read_csv(DISEASE_DATASET_PATH)
    except UnicodeDecodeError:
        disease_df = pd.read_csv(DISEASE_DATASET_PATH, encoding='latin-1')
    
    # Function to clean DataFrame column names for consistent lookup
    def clean_df_column_name_for_lookup(name):
        return re.sub(r'[^a-zA-Z0-9]+', '_', str(name)).strip('_').lower()
    
    disease_df.columns = [clean_df_column_name_for_lookup(col) for col in disease_df.columns]

    print("Local disease classifier, symptom columns, and dataset loaded successfully.")
    DISEASE_MODEL_LOADED = True
except (FileNotFoundError, pickle.UnpackingError, pd.errors.EmptyDataError) as e:
    print(f"WARNING: Could not load local disease model files or dataset. Reason: {e}. Disease prediction will fall back to LLM APIs.")

STUDY_GUIDE_LOADED = False
# Attempt to load the local study guide dataset
try:
    study_guide_df = pd.read_csv(STUDY_GUIDE_DATASET_PATH, encoding='utf-8')
    def clean_study_guide_col_name(name):
        return re.sub(r'[^a-zA-Z0-9]+', '_', str(name)).strip('_').lower()
    study_guide_df.columns = [clean_study_guide_col_name(col) for col in study_guide_df.columns]
    print("Local study guide dataset loaded successfully.")
    STUDY_GUIDE_LOADED = True
except (FileNotFoundError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
    print(f"WARNING: Could not load local study guide dataset. Reason: {e}. Study guidance will fall back to LLM APIs.")

# Attempt to load the local sentiment classifier
SENTIMENT_MODEL_LOADED = False
try:
    with open(SENTIMENT_VECTORIZER_PATH, 'rb') as f:
        sentiment_vectorizer = pickle.load(f)
    with open(SENTIMENT_CLASSIFIER_PATH, 'rb') as f:
        sentiment_classifier = pickle.load(f)
    print("Local sentiment classifier loaded successfully.")
    SENTIMENT_MODEL_LOADED = True
except (FileNotFoundError, pickle.UnpackingError) as e:
    print(f"WARNING: Could not load local sentiment model files. Reason: {e}. Sentiment analysis will fall back to LLM APIs.")

## Attempt to load the local code analysis dataset
CODE_ANALYSIS_LOADED = False
try:
    code_analysis_df = pd.read_csv(CODE_ANALYSIS_DATASET_PATH, encoding='utf-8')
    def clean_code_analysis_col_name(name):
        return re.sub(r'[^a-zA-Z0-9]+', '_', str(name)).strip('_').lower()
    code_analysis_df.columns = [clean_code_analysis_col_name(col) for col in code_analysis_df.columns]
    print("Local code analysis dataset loaded successfully.")
    CODE_ANALYSIS_LOADED = True
except (FileNotFoundError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
    print(f"WARNING: Could not load local code analysis dataset. Reason: {e}. Code analysis will fall back to LLM APIs.")

## Code Generation Dataset
CODE_GENERATION_LOADED = False
try:
    code_generation_df = pd.read_csv(CODE_GENERATION_DATASET_PATH, encoding='utf-8')
    print("Local code generation dataset loaded successfully.")
    CODE_GENERATION_LOADED = True
except (FileNotFoundError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
    print(f"WARNING: Could not load local code generation dataset. Reason: {e}. Code generation will fall back to LLM APIs.")

# Attempt to load the local fake news classifier and dataset
FAKE_NEWS_LOADED = False
try:
    with open(FAKE_NEWS_VECTORIZER_PATH, 'rb') as f:
        fake_news_vectorizer = pickle.load(f)
    with open(FAKE_NEWS_CLASSIFIER_PATH, 'rb') as f:
        fake_news_classifier = pickle.load(f)
    fake_news_df = pd.read_csv(FAKE_NEWS_DATASET_PATH, encoding='utf-8')
    print("Local fake news classifier and dataset loaded successfully.")
    FAKE_NEWS_LOADED = True
except (FileNotFoundError, pickle.UnpackingError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
    print(f"WARNING: Could not load local fake news model files or dataset. Reason: {e}. Fake news detection will fall back to LLM APIs.")

# Attempt to load the local interview Q&A dataset
INTERVIEW_QA_LOADED = False
try:
    interview_qa_df = pd.read_csv(INTERVIEW_QA_DATASET_PATH, encoding='utf-8')
    def clean_interview_qa_col_name(name):
        return re.sub(r'[^a-zA-Z0-9]+', '_', str(name)).strip('_').lower()
    interview_qa_df.columns = [clean_interview_qa_col_name(col) for col in interview_qa_df.columns]
    print("Local interview Q&A dataset loaded successfully.")
    INTERVIEW_QA_LOADED = True
except (FileNotFoundError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
    print(f"WARNING: Could not load local interview Q&A dataset. Reason: {e}. Interview Q&A will fall back to LLM APIs.")

# Attempt to load the local plagiarism corpus dataset
PLAGIARISM_CORPUS_LOADED = False # Attempt to load the local plagiarism corpus dataset
try:
    plagiarism_corpus_df = pd.read_csv(PLAGIARISM_CORPUS_PATH)
    print("Local plagiarism corpus loaded successfully.")
    PLAGIARISM_CORPUS_LOADED = True
except Exception as e:
    print(f"WARNING: Could not load local plagiarism corpus: {e}")
# If the corpus is not loaded, plagiarism detection will rely on LLM APIs.

# Attempt to load the local recipe dataset
RECIPE_DATASET_LOADED = False
try:
    recipe_df = pd.read_csv(RECIPE_DATASET_PATH, encoding='utf-8')
    print("Local recipe dataset loaded successfully.")
    RECIPE_DATASET_LOADED = True
except (FileNotFoundError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
    print(f"WARNING: Could not load local recipe dataset. Reason: {e}. Recipe generation will fall back to LLM APIs.")



# --- LLM API Wrapper Functions ---

def call_gemini_api(prompt, response_schema=None, audio_config=None):
    """Makes a call to the Gemini API."""
    api_key = API_KEYS.get("gemini")
    if not api_key:
        print("ERROR: Gemini API key not set.")
        return None

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    
    # Determine which Gemini endpoint to use
    url = f"{API_ENDPOINTS['gemini']}?key={api_key}"
    if audio_config:
        url = f"{API_ENDPOINTS['gemini_tts']}?key={api_key}"
        payload["generationConfig"] = {
            "responseModalities": ["AUDIO"],
            "speechConfig": audio_config
        }
    elif response_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60) # Increased timeout for potentially longer summarization/fetching
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        if audio_config:
            # Handle audio response
            part = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0]
            audio_data = part.get('inlineData', {}).get('data')
            mime_type = part.get('inlineData', {}).get('mimeType')
            if audio_data and mime_type and mime_type.startswith("audio/"):
                # Extract sample rate from mimeType, e.g., "audio/L16;rate=16000"
                sample_rate_match = re.search(r'rate=(\d+)', mime_type)
                sample_rate = int(sample_rate_match.group(1)) if sample_rate_match else 16000 # Default to 16kHz
                return {'audio_data': audio_data, 'mime_type': mime_type, 'sample_rate': sample_rate}
            return None
        elif 'candidates' in result and len(result['candidates']) > 0:
            text_response = result['candidates'][0]['content']['parts'][0]['text']
            if response_schema:
                try:
                    return json.loads(text_response)
                except json.JSONDecodeError:
                    print(f"WARNING: Gemini returned non-JSON for schema request: {text_response[:100]}...")
                    return None
            return text_response
        return None
    except requests.exceptions.RequestException as e:
        print(f"Gemini API call failed: {e}")
        return None

def call_openai_api(prompt, response_schema=None, audio_config=None):
    """Makes a call to the OpenAI API."""
    # OpenAI does not have a direct TTS API through this endpoint.
    # If audio_config is requested, this function will not handle it.
    if audio_config:
        print("WARNING: OpenAI API does not support direct TTS via this chat completions endpoint.")
        return None

    api_key = API_KEYS.get("openai")
    if not api_key:
        print("ERROR: OpenAI API key not set.")
        return None
    
    payload = {
        "model": "gpt-3.5-turbo", # Example model, can be configured
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000 # Increased max_tokens for summarization
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    if response_schema:
        payload["messages"][0]["content"] += "\n\nRespond only in JSON format matching this structure: " + json.dumps(response_schema)
        
    url = API_ENDPOINTS['openai']
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60) # Increased timeout
        response.raise_for_status()
        result = response.json()
        if result and 'choices' in result and result['choices'] and 'message' in result['choices'][0]:
            text_response = result['choices'][0]['message']['content']
            if response_schema:
                try:
                    return json.loads(text_response)
                except json.JSONDecodeError:
                    print(f"WARNING: OpenAI returned non-JSON for schema request: {text_response[:100]}...")
                    return None
            return text_response
        return None
    except requests.exceptions.RequestException as e:
        print(f"OpenAI API call failed: {e}")
        return None

def call_groq_api(prompt, response_schema=None, audio_config=None):
    """Makes a call to the Groq API (OpenAI-compatible)."""
    if audio_config:
        print("WARNING: Groq API does not support direct TTS via this chat completions endpoint.")
        return None

    api_key = API_KEYS.get("groq")
    if not api_key:
        print("ERROR: Groq API key not set.")
        return None
    
    payload = {
        "model": "llama3-8b-8192", # Example Groq model, can be configured
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000 # Increased max_tokens
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    if response_schema:
        payload["messages"][0]["content"] += "\n\nRespond only in JSON format matching this structure: " + json.dumps(response_schema)

    url = API_ENDPOINTS['groq']
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60) # Increased timeout
        response.raise_for_status()
        result = response.json()
        if result and 'choices' in result and result['choices'] and 'message' in result['choices'][0]:
            text_response = result['choices'][0]['message']['content']
            if response_schema:
                try:
                    return json.loads(text_response)
                except json.JSONDecodeError:
                    print(f"WARNING: Groq returned non-JSON for schema request: {text_response[:100]}...")
                    return None
            return text_response
        return None
    except requests.exceptions.RequestException as e:
        print(f"Groq API call failed: {e}")
        return None

def call_microsoft_phi3_api(prompt, response_schema=None, audio_config=None):
    """Makes a call to a hypothetical Microsoft Phi-3 API."""
    if audio_config:
        print("WARNING: Microsoft Phi-3 API does not support direct TTS via this endpoint.")
        return None

    api_key = API_KEYS.get("microsoft_phi3")
    if not api_key:
        print("ERROR: Microsoft Phi-3 API key not set.")
        return None
    
    payload = {
        "prompt": prompt,
        "max_new_tokens": 1000 # Increased max_new_tokens
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}' # Or other authentication scheme
    }

    if response_schema:
        payload["prompt"] += "\n\nRespond only in JSON format matching this structure: " + json.dumps(response_schema)

    url = API_ENDPOINTS['microsoft_phi3']
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60) # Increased timeout
        response.raise_for_status()
        result = response.json()
        # Adjust based on actual Phi-3 API response structure
        if result and 'generated_text' in result:
            text_response = result['generated_text']
            if response_schema:
                try:
                    return json.loads(text_response)
                except json.JSONDecodeError:
                    print(f"WARNING: Phi-3 returned non-JSON for schema request: {text_response[:100]}...")
                    return None
            return text_response
        return None
    except requests.exceptions.RequestException as e:
        print(f"Microsoft Phi-3 API call failed: {e}")
        return None

def call_tencent_hunyuan_api(prompt, response_schema=None, audio_config=None):
    """Makes a call to a hypothetical Tencent Hunyuan-A13B-Instruct API."""
    if audio_config:
        print("WARNING: Tencent Hunyuan API does not support direct TTS via this endpoint.")
        return None

    api_key = API_KEYS.get("tencent_hunyuan")
    if not api_key:
        print("ERROR: Tencent Hunyuan API key not set.")
        return None
    
    payload = {
        "input": prompt,
        "parameters": {"max_new_tokens": 1000} # Increased max_new_tokens
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}' # Or other authentication scheme
    }

    if response_schema:
        payload["input"] += "\n\nRespond only in JSON format matching this structure: " + json.dumps(response_schema)

    url = API_ENDPOINTS['tencent_hunyuan']
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60) # Increased timeout
        response.raise_for_status()
        result = response.json()
        # Adjust based on actual Tencent API response structure
        if result and 'output' in result:
            text_response = result['output']
            if response_schema:
                try:
                    return json.loads(text_response)
                except json.JSONDecodeError:
                    print(f"WARNING: Tencent Hunyuan returned non-JSON for schema request: {text_response[:100]}...")
                    return None
            return text_response
        return None
    except requests.exceptions.RequestException as e:
        print(f"Tencent Hunyuan API call failed: {e}")
        return None

def call_copilot_api(prompt, response_schema=None, audio_config=None):
    """Makes a call to a hypothetical Copilot API."""
    if audio_config:
        print("WARNING: Copilot API does not support direct TTS via this endpoint.")
        return None

    api_key = API_KEYS.get("copilot")
    if not api_key:
        print("ERROR: Copilot API key not set.")
        return None
    
    payload = {
        "query": prompt,
        "max_tokens": 1000 # Increased max_tokens
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}' # Or other authentication scheme
    }

    if response_schema:
        payload["query"] += "\n\nRespond only in JSON format matching this structure: " + json.dumps(response_schema)

    url = API_ENDPOINTS['copilot']
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60) # Increased timeout
        response.raise_for_status()
        result = response.json()
        if result and 'response' in result:
            text_response = result['response']
            if response_schema:
                try:
                    return json.loads(text_response)
                except json.JSONDecodeError:
                    print(f"WARNING: Copilot returned non-JSON for schema request: {text_response[:100]}...")
                    return None
            return text_response
        return None
    except requests.exceptions.RequestException as e:
        print(f"Copilot API call failed: {e}")
        return None


def orchestrate_llm_calls(prompt, response_schema=None, audio_config=None):
    """
    Orchestrates calls to multiple LLM APIs in a defined priority order.
    Returns the first satisfactory response (not None and not empty) or None if all fail.
    Can now also handle structured JSON responses using response_schema or audio data.
    """
    llm_call_functions = {
        "gemini": call_gemini_api,
        "openai": call_openai_api,
        "groq": call_groq_api,
        "microsoft_phi3": call_microsoft_phi3_api,
        "tencent_hunyuan": call_tencent_hunyuan_api,
        "copilot": call_copilot_api
    }

    # If audio_config is present, prioritize Gemini for TTS
    if audio_config:
        print("Attempting TTS via Gemini API...")
        tts_response = call_gemini_api(prompt, audio_config=audio_config)
        if tts_response:
            return tts_response
        print("Gemini TTS failed. No other LLM supports direct audio generation in this setup.")
        return None # No fallback for audio if Gemini fails

    # For text/JSON responses, proceed with general orchestration
    for llm_name in LLM_PRIORITY_ORDER:
        print(f"Attempting API call with {llm_name}...")
        try:
            response = llm_call_functions[llm_name](prompt, response_schema=response_schema)
            if response:
                if response_schema and isinstance(response, (dict, list)):
                    print(f"Successfully got structured response from {llm_name}.")
                    return response
                elif not response_schema and isinstance(response, str) and response.strip():
                    print(f"Successfully got text response from {llm_name}.")
                    return response
            else:
                print(f"{llm_name} returned empty or unsatisfactory response. Trying next API.")
        except Exception as e:
            print(f"Error calling {llm_name}: {e}. Trying next API.")
    
    return None # All LLMs failed to provide a satisfactory response


# --- Language Detection and Translation Functions using LLM Orchestration ---

def detect_language_llm(text):
    """Detects the language of the given text using LLM orchestration."""
    try:
        # Prompt the LLM to detect language and return ISO 639-1 code
        prompt = f"Detect the language of the following text and respond only with the ISO 639-1 language code (e.g., 'en', 'hi', 'bn'). If unsure, respond with 'en'.\n\nText: \"{text}\""
        lang_code = orchestrate_llm_calls(prompt)
        if lang_code:
            # Clean up the response to ensure it's just the 2-letter code
            lang_code = str(lang_code).strip().lower().split('\n')[0].replace('language code: ', '').replace('iso 639-1: ', '')
            if len(lang_code) == 2 and lang_code.isalpha():
                return lang_code
        return "en" # Default to English if LLM detection is unclear
    except Exception as e:
        print(f"WARNING: LLM language detection failed: {e}. Defaulting to 'en'.")
        return "en"

def detect_user_language(text):
    """
    Detects the language of the user's input.
    Prioritizes common hardcoded greetings, then local 'langdetect', then LLM orchestration.
    """
    text_lower = text.lower()

    # 1. Handle common English greetings directly
    common_english_greetings = ["hi", "hello", "hey", "hii", "hlo"]
    if text_lower in common_english_greetings:
        print(f"DEBUG: '{text}' directly identified as English (Common English Greeting).")
        return "en"

    # 2. Handle common Bengali greetings directly
    # Use actual Bengali script for matching
    common_bengali_greetings = ["নমস্কার", "কেমন আছেন", "শুভ সকাল", "হ্যালো"]
    if text == "নমস্কার" or any(greeting in text for greeting in common_bengali_greetings):
        print(f"DEBUG: '{text}' directly identified as Bengali (Common Bengali Greeting).")
        return "bn" # Return Bengali language code

    # 3. Attempt local 'langdetect' if available
    if LANGDETECT_AVAILABLE:
        try:
            detected_lang = detect(text)
            print(f"DEBUG: Language detected locally (langdetect): {detected_lang}")
            return detected_lang
        except Exception as e:
            print(f"WARNING: Local langdetect failed for '{text}': {e}. Falling back to LLM orchestration for detection.")
    
    # 4. Fallback to LLM orchestration for language detection
    llm_detected_lang = detect_language_llm(text)
    print(f"DEBUG: Language detected using LLM orchestration: {llm_detected_lang}")
    return llm_detected_lang

def translate_text_llm(text, target_language, source_language=None):
    """Translates text using LLM orchestration."""
    try:
        if source_language:
            prompt = f"Translate the following text from {source_language} to {target_language}. Provide only the translated text, no extra commentary:\n\nText: \"{text}\""
        else:
            prompt = f"Translate the following text to {target_language}. Provide only the translated text, no extra commentary:\n\nText: \"{text}\""
        
        translated_text = orchestrate_llm_calls(prompt)
        if translated_text:
            translated_text = str(translated_text).strip()
            # Remove surrounding quotes if the LLM adds them
            if translated_text.startswith('"') and translated_text.endswith('"'):
                translated_text = translated_text[1:-1]
            print(f"DEBUG: Translated using LLM orchestration: {translated_text}")
            return translated_text
        return text # Return original text if translation fails or returns empty
    except Exception as e:
        print(f"WARNING: LLM translation failed: {e}. Returning original text.")
        return text

def extract_location_from_text(text):
    """
    Attempts to extract a location (city, country, or region) from the user's text
    using LLM orchestration.
    """
    try:
        prompt = f"""
        Analyze the following text and extract any mentioned geographical location (city, state, country, or region).
        Respond ONLY with the location name, or 'None' if no clear location is mentioned.
        Examples:
        Text: "Looking for jobs in Bangalore, India." -> Bangalore, India
        Text: "Best universities in the UK." -> UK
        Text: "Symptoms in my area." -> None
        Text: "How to study for data science in California?" -> California
        Text: "What's the job market like in Germany?" -> Germany
        Text: "Tell me about diseases." -> None
        
        Text: "{text}"
        """
        location = orchestrate_llm_calls(prompt)
        if location:
            location = str(location).strip()
            if location.lower() == 'none' or not location:
                return None
            return location
        return None
    except Exception as e:
        print(f"WARNING: Failed to extract location from text using LLM orchestration: {e}")
        return None

def extract_code_and_requirement(text):
    """
    Extracts code snippet and user's requirement from a text input.
    Handles code enclosed in triple backticks (```) or single backticks (`).
    """
    code_pattern_triple = r"```(?P<language>\w+)?\n(?P<code_triple>[\s\S]+?)\n```"
    code_pattern_single = r"`(?P<code_single>[^`]+?)`"

    code_snippet = None
    remaining_text = text

    match_triple = re.search(code_pattern_triple, text)
    if match_triple:
        code_snippet = match_triple.group('code_triple').strip()
        remaining_text = text.replace(match_triple.group(0), '').strip()
        print(f"DEBUG: Extracted code snippet (triple backticks): {code_snippet[:50]}...")
        return code_snippet, remaining_text

    match_single = re.search(code_pattern_single, text)
    if match_single:
        code_snippet = match_single.group('code_single').strip()
        remaining_text = text.replace(match_single.group(0), '').strip()
        print(f"DEBUG: Extracted code snippet (single backticks): {code_snippet[:50]}...")
        return code_snippet, remaining_text
    
    # Fallback for code snippets using a colon, e.g., "Python code: print('Hello')"
    if ':' in text:
        parts = text.split(':', 1)
        if len(parts) > 1 and len(parts[1].strip()) > 5: # Ensure there's substantial text after the colon
            code_snippet = parts[1].strip()
            remaining_text = parts[0].strip()
            print(f"DEBUG: Extracted code (colon method): {code_snippet[:50]}...")
            return code_snippet, remaining_text
    
    return code_snippet, remaining_text

def extract_url_or_content_for_fake_news(text):
    """
    Extracts a URL or the main text content to be analyzed for fake news.
    """
    url_pattern = r"https?://[^\s]+"
    url_match = re.search(url_pattern, text)
    
    if url_match:
        url = url_match.group(0)
        remaining_query = text.replace(url, '').strip()
        print(f"DEBUG: Detected URL for fake news analysis: {url}")
        return url, remaining_query, "url"
    else:
        print("DEBUG: No URL detected. Analyzing text content for fake news.")
        return text, text, "text"

def extract_url_and_command(text):
    """
    Extracts a URL and a quoted command from the user's message.
    Example: "[https://example.com/video.mp4](https://example.com/video.mp4) "download video""
    """
    # Regex to find a URL at the beginning, followed by optional space, then a quoted string
    match = re.match(r'^(https?://[^\s]+)\s*"(.*?)"$', text.strip())
    if match:
        url = match.group(1)
        command = match.group(2).lower().strip()
        print(f"DEBUG: Extracted URL: {url}, Command: {command}")
        return url, command
    return None, None



def extract_data_and_command(text, keywords):
    """
    Extracts the data (to encode) and the command (in double quotes) from user input.
    Returns (data, command) or (None, None) if not matched.
    Example: 'https://github.com "generate qr code"' -> ('https://github.com', 'generate qr code')
    """
    match = re.match(r'^(.*?)\s*"([^"]+)"\s*$', text.strip())
    if match:
        data = match.group(1).strip()
        command = match.group(2).strip().lower()
        # Only return if command matches a keyword
        if any(kw in command for kw in keywords):
            return data, command
    return None, None



def handle_media_task(url, command):
    """
    Handles media download/extraction tasks using yt-dlp.
    Returns a dictionary with 'type' (e.g., 'download_link', 'text') and 'content' or 'file_path' and 'filename'.
    Also includes 'temp_dir' for cleanup.
    """
    # Create a temporary directory for downloads
    temp_dir = tempfile.mkdtemp()
    print(f"DEBUG: Created temporary directory: {temp_dir}")
    
    try:
        ydl_opts_info = {
            'quiet': True,
            'skip_download': True,
            'force_generic_extractor': True, # Try to extract info from non-standard URLs
            'no_warnings': True,
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info_dict = ydl.extract_info(url, download=False) # Get video info without downloading
            
        if not info_dict:
            return {'type': 'error', 'content': "Could not retrieve video information from the provided URL.", 'temp_dir': temp_dir}

        # Clean up title for filename
        title = info_dict.get('title', 'downloaded_file')
        # Remove characters that are problematic in filenames
        safe_title = re.sub(r'[^\w\s.-]', '', title).strip()
        safe_title = re.sub(r'\s+', '_', safe_title) # Replace spaces with underscores

        if "download video" in command or command == "download":
            # Logic for "mid quality" video download
            formats = info_dict.get('formats', [])
            video_formats = [f for f in formats if f.get('vcodec') != 'none' and f.get('acodec') != 'none']
            
            # Sort by height (quality)
            video_formats.sort(key=lambda x: x.get('height', 0), reverse=True)

            selected_format = None
            if len(video_formats) >= 3:
                # Select the second highest quality (mid quality)
                selected_format = video_formats[1]
            elif len(video_formats) == 2:
                # If only two, pick the lower one as "mid"
                selected_format = video_formats[1]
            elif len(video_formats) == 1:
                selected_format = video_formats[0]
            
            if selected_format:
                ydl_opts_video = {
                    'format': selected_format['format_id'],
                    'outtmpl': os.path.join(temp_dir, f'{safe_title}.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
                    ydl.download([url])
                
                # Find the actual downloaded file (yt-dlp might change extension)
                downloaded_file_path = None
                for f_name in os.listdir(temp_dir):
                    if f_name.startswith(safe_title):
                        downloaded_file_path = os.path.join(temp_dir, f_name)
                        break
                
                if downloaded_file_path:
                    download_filename = os.path.basename(downloaded_file_path)
                    print(f"DEBUG: Video downloaded to: {downloaded_file_path}")
                    return {'type': 'download_link', 'file_path': downloaded_file_path, 'filename': download_filename, 'temp_dir': temp_dir}
                else:
                    return {'type': 'error', 'content': "Video download failed: No file found after download.", 'temp_dir': temp_dir}
            else:
                return {'type': 'error', 'content': "No suitable video format found for download.", 'temp_dir': temp_dir}

        elif "audio download" in command:
            ydl_opts_audio = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3', # Or 'm4a', 'wav'
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(temp_dir, f'{safe_title}.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                ydl.download([url])
            
            downloaded_file_path = None
            # yt-dlp might save as .webm or .m4a then convert to .mp3, so check for .mp3 first
            for f_name in os.listdir(temp_dir):
                if f_name.startswith(safe_title) and f_name.endswith('.mp3'):
                    downloaded_file_path = os.path.join(temp_dir, f_name)
                    break
            if not downloaded_file_path: # Fallback if not mp3 but still downloaded
                 for f_name in os.listdir(temp_dir):
                    if f_name.startswith(safe_title):
                        downloaded_file_path = os.path.join(temp_dir, f_name)
                        break

            if downloaded_file_path:
                download_filename = os.path.basename(downloaded_file_path)
                print(f"DEBUG: Audio downloaded to: {downloaded_file_path}")
                return {'type': 'download_link', 'file_path': downloaded_file_path, 'filename': download_filename, 'temp_dir': temp_dir}
            else:
                return {'type': 'error', 'content': "Audio download failed: No file found.", 'temp_dir': temp_dir}

        elif "download thumbnail" in command:
            thumbnail_url = info_dict.get('thumbnail')
            if thumbnail_url:
                try:
                    thumb_response = requests.get(thumbnail_url, stream=True)
                    thumb_response.raise_for_status()
                    thumb_ext = thumbnail_url.split('.')[-1].split('?')[0] # Get extension from URL
                    if len(thumb_ext) > 4 or '/' in thumb_ext: # Basic check for valid extension
                        thumb_ext = 'jpg' # Default if extension is messy
                    thumbnail_filename = f"{safe_title}_thumbnail.{thumb_ext}"
                    thumbnail_path = os.path.join(temp_dir, thumbnail_filename)
                    with open(thumbnail_path, 'wb') as f:
                        for chunk in thumb_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"DEBUG: Thumbnail downloaded to: {thumbnail_path}")
                    return {'type': 'download_link', 'file_path': thumbnail_path, 'filename': thumbnail_filename, 'temp_dir': temp_dir}
                except requests.exceptions.RequestException as e:
                    return {'type': 'error', 'content': f"Failed to download thumbnail: {e}", 'temp_dir': temp_dir}
            else:
                return {'type': 'error', 'content': "No thumbnail found for this video.", 'temp_dir': temp_dir}

        elif "extract video script" in command:
            subtitles = info_dict.get('subtitles') or info_dict.get('automatic_captions')
            script_text = ""
            if subtitles:
                # Try to get English subtitles first
                en_subs = subtitles.get('en') or subtitles.get('en-US')
                if en_subs:
                    sub_url = en_subs[0]['url']
                    sub_format = en_subs[0]['ext']
                    try:
                        sub_content = requests.get(sub_url).text
                        # Basic parsing for .vtt or .srt to extract text
                        if sub_format == 'vtt':
                            lines = sub_content.split('\n')
                            for line in lines:
                                if not line.strip() or "WEBVTT" in line or "-->" in line or line.isdigit():
                                    continue
                                script_text += line.strip() + " "
                        elif sub_format == 'srt':
                            lines = sub_content.split('\n')
                            for line in lines:
                                if not line.strip() or "-->" in line or line.isdigit():
                                    continue
                                script_text += line.strip() + " "
                        else:
                            script_text = sub_content # Fallback for other formats
                        
                        script_text = script_text.strip()
                        if not script_text:
                            return {'type': 'error', 'content': "Extracted script was empty.", 'temp_dir': temp_dir}

                        # Save the script to a .txt file
                        script_filename = f"{safe_title}_script.txt"
                        script_path = os.path.join(temp_dir, script_filename)
                        with open(script_path, 'w', encoding='utf-8') as f:
                            f.write(script_text)
                        
                        print(f"DEBUG: Script extracted and saved to: {script_path}")
                        return {'type': 'download_link', 'file_path': script_path, 'filename': script_filename, 'temp_dir': temp_dir}

                    except requests.exceptions.RequestException as e:
                        return {'type': 'error', 'content': f"Failed to download subtitles: {e}", 'temp_dir': temp_dir}
                else:
                    return {'type': 'error', 'content': "No English subtitles or captions found for this video.", 'temp_dir': temp_dir}
            else:
                return {'type': 'error', 'content': "No subtitles or captions found for this video.", 'temp_dir': temp_dir}


        elif "extract video tags" in command:
            tags = info_dict.get('tags')
            if tags:
                return {'type': 'text', 'content': f"Video Tags: {', '.join(tags)}", 'temp_dir': temp_dir}
            else:
                return {'type': 'text', 'content': "No tags found for this video.", 'temp_dir': temp_dir}

        elif "extract video views and earnings" in command:
            # This is generally NOT possible via yt-dlp or simple web scraping.
            # Views might be present in info_dict, but earnings are private.
            views = info_dict.get('view_count')
            response_text = "Extracting video earnings is not publicly possible as it's private creator data. "
            if views:
                response_text += f"However, the video has approximately {views:,} views."
            else:
                response_text += "Views count could not be retrieved."
            
            # Fallback to LLM to explain why earnings cannot be extracted
            llm_explanation_prompt = f"Explain why it's generally not possible to extract video earnings from a public video link. Also, mention if views count is usually available."
            llm_explanation = orchestrate_llm_calls(llm_explanation_prompt)
            
            if llm_explanation:
                return {'type': 'text', 'content': f"**Regarding your request to extract views and earnings:**\n\n{response_text}\n\nHere's why extracting earnings is typically not possible:\n{llm_explanation}", 'temp_dir': temp_dir}
            else:
                return {'type': 'text', 'content': response_text + "\n\nI couldn't get a detailed explanation on why earnings are not available, but it's generally private information.", 'temp_dir': temp_dir}

        else:
            return {'type': 'error', 'content': "Unsupported media command.", 'temp_dir': temp_dir}

    except yt_dlp.DownloadError as e:
        print(f"ERROR: yt-dlp download error: {e}")
        return {'type': 'error', 'content': f"Failed to process video link: {e}. The URL might be unsupported or the video unavailable.", 'temp_dir': temp_dir}
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during media handling: {e}")
        return {'type': 'error', 'content': f"An unexpected error occurred while processing your request: {e}. Please try again or check the URL.", 'temp_dir': temp_dir}
    finally:
        # The temporary directory needs to persist until the file is served.
        # So, we don't delete it here immediately for 'download_link' types.
        # Cleanup will happen in the /download_media endpoint after send_file.
        # For 'text' or 'error' types, we can clean up immediately.
        pass # The cleanup logic is moved to the chat function's main try-except-finally


















# SQLite DB path (local)
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "db/users.db")

# MongoDB Atlas connection (for sessions/logs) - set MONGO_URI in .env
MONGO_URI = os.getenv("MONGO_URI", "")
mongo_client = None
mongo_db = None
if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_db = mongo_client.get_database()  # default db from URI
        print("Connected to MongoDB Atlas.")
    except Exception as e:
        print(f"WARNING: Could not connect to MongoDB Atlas: {e}")
else:
    print("WARNING: MONGO_URI not set. Sessions will be stored in-memory (development only).")

# In-memory session store fallback if Mongo not available
_in_memory_sessions = {}

# Ensure SQLite users table exists
def init_sqlite_db():
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    # cur.execute("DROP TABLE IF EXISTS users")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT,
        dob TEXT,
        username TEXT UNIQUE,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at TEXT
    )
    """)
    conn.commit()
    return conn

sqlite_conn = init_sqlite_db()
sqlite_cursor = sqlite_conn.cursor()

# Helper functions for SQLite users
def create_user_sqlite(email, password_hash, role="user",
                       full_name=None, dob=None, username=None):
    try:
        created_at = datetime.utcnow().isoformat()
        sqlite_cursor.execute(
            """INSERT INTO users (full_name, dob, username, email, password_hash, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (full_name, dob, username, email, password_hash, role, created_at)
        )
        sqlite_conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        return False, "Email or Username already exists."
    except Exception as e:
        return False, str(e)

def find_user_by_email(email):
    sqlite_cursor.execute("SELECT id, email, password_hash, role FROM users WHERE email = ?", (email,))
    row = sqlite_cursor.fetchone()
    if row:
        return {"id": row[0], "email": row[1], "password_hash": row[2], "role": row[3]}
    return None

# Session management (stored in MongoDB Atlas if available, otherwise in-memory)
def create_session(email, role, ttl_minutes=120):
    token = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
    session_doc = {
        "token": token,
        "email": email,
        "role": role,
        "created_at": datetime.utcnow(),
        "expires_at": expires_at
    }
    if mongo_db is not None:
        try:
            mongo_db.sessions.insert_one(session_doc)
        except Exception as e:
            print("WARNING: Failed to write session to Mongo:", e)
            _in_memory_sessions[token] = session_doc
    else:
        _in_memory_sessions[token] = session_doc
    return token, expires_at

def get_session(token):
    if not token:
        return None
    if mongo_db:
        s = mongo_db.sessions.find_one({"token": token})
        if s:
            if s.get("expires_at") and s["expires_at"] < datetime.utcnow():
                # expired
                mongo_db.sessions.delete_one({"token": token})
                return None
            return s
        return None
    else:
        s = _in_memory_sessions.get(token)
        if not s:
            return None
        if s["expires_at"] < datetime.utcnow():
            del _in_memory_sessions[token]
            return None
        return s

def destroy_session(token):
    if not token:
        return False
    if mongo_db:
        mongo_db.sessions.delete_one({"token": token})
    else:
        if token in _in_memory_sessions: del _in_memory_sessions[token]
    return True

def log_admin_action(admin_email, action, details=None):
    doc = {
        "admin": admin_email,
        "action": action,
        "details": details,
        "timestamp": datetime.utcnow()
    }
    if mongo_db:
        try:
            mongo_db.admin_logs.insert_one(doc)
        except Exception as e:
            print("WARNING: Failed to log admin action to Mongo:", e)
    else:
        print("ADMIN LOG:", doc)

# --- Helper decorator for token-protected endpoints ---
from functools import wraps
def require_auth(role_allowed=None):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            token = request.headers.get("X-Auth-Token") or request.form.get("token") or request.args.get("token")
            if not token:
                return jsonify({"error": "Authentication token required."}), 401
            session = get_session(token)
            if not session:
                return jsonify({"error": "Invalid or expired token."}), 401
            if role_allowed and session.get("role") not in role_allowed:
                return jsonify({"error": "Insufficient privileges."}), 403
            # attach session info to request context if needed
            request._session = session
            return f(*args, **kwargs)
        return wrapped
    return decorator









@app.route("/add_user", methods=["POST"])
def add_user():
    data = request.json
    mongo.db.users.insert_one(data)   # <-- use mongo here
    return jsonify({"message": "User added!"})









# --- NEW: Register endpoint ---
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()

    full_name = data.get("full_name", "").strip()
    dob = data.get("dob", "").strip()
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    # Validate all fields
    if not all([full_name, dob, username, email, password]):
        return jsonify({"error": "All fields are required"}), 400

    # Check duplicates (username & email)
    if mongo.db.users.find_one({"email": email}) or mongo.db.users.find_one({"username": username}):
        return jsonify({"error": "Email or Username already exists"}), 400

    # Save to MongoDB
    user_doc = {
        "full_name": full_name,
        "dob": dob,
        "username": username,
        "email": email,
        "password": generate_password_hash(password),
        "role": "user"
    }
    mongo.db.users.insert_one(user_doc)

    # Save to SQLite (if you're mirroring data)
    ok, err = create_user_sqlite(
        email=email,
        password_hash=generate_password_hash(password),
        role="user",
        full_name=full_name,
        dob=dob,
        username=username
    )
    if not ok:
        return jsonify({"error": err}), 400

    return jsonify({"success": True, "message": "Registered successfully"}), 201






# --- NEW: Login endpoint (returns token) ---
@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.form if request.form else (request.json or {})
        email = data.get("email")
        password = data.get("password")
        if not email or not password:
            return jsonify({"error": "Email and password required."}), 400
        user = find_user_by_email(email.lower().strip())
        if not user:
            return jsonify({"error": "Invalid credentials."}), 401
        if not checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
            return jsonify({"error": "Invalid credentials."}), 401
        token, expires_at = create_session(user["email"], user["role"])
        # Return token + role + email
        return jsonify({"token": token, "expires_at": expires_at.isoformat(), "role": user["role"], "email": user["email"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NEW: Logout endpoint ---
@app.route("/logout", methods=["POST"])
@require_auth()
def logout():
    token = request.headers.get("X-Auth-Token") or (request.form.get("token"))
    if not token:
        return jsonify({"error": "No token provided."}), 400
    destroyed = destroy_session(token)
    return jsonify({"success": destroyed})

# --- NEW: Admin creation endpoint (admin-only) ---
@app.route("/admin/create", methods=["POST"])
@require_auth(role_allowed=["admin"])
def admin_create():
    try:
        session = request._session
        admin_email = session.get("email")
        data = request.form if request.form else (request.json or {})
        new_email = data.get("email")
        new_password = data.get("password")
        new_role = data.get("role") or "admin"
        if not new_email or not new_password:
            return jsonify({"error": "email and password required."}), 400
        pw_hash = hashpw(new_password.encode("utf-8"), gensalt()).decode("utf-8")
        ok, err = create_user_sqlite(new_email.lower().strip(), pw_hash, role=new_role)
        if not ok:
            return jsonify({"error": err}), 400
        log_admin_action(admin_email, "create_admin", {"created": new_email})
        return jsonify({"success": True, "message": f"Admin {new_email} created."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NEW: Admin list users endpoint (admin-only) ---
@app.route("/admin/list_users", methods=["GET"])
@require_auth(role_allowed=["admin"])
def admin_list_users():
    try:
        sqlite_cursor.execute("SELECT id, email, role, created_at FROM users ORDER BY id DESC")
        rows = sqlite_cursor.fetchall()
        users = [{"id": r[0], "email": r[1], "role": r[2], "created_at": r[3]} for r in rows]
        return jsonify({"users": users})
    except Exception as e:
        return jsonify({"error": str(e)}), 500






# Global dictionary to store temporary file paths, keyed by a unique ID
# This is a simple in-memory map. For production, consider a more persistent/scalable solution.
temp_file_map = {}

@app.route('/download_media/<unique_id>', methods=['GET'])

def download_media(unique_id):
    """
    Serves a temporary file to the user for download and then deletes it.
    The unique_id maps to the file's path and original download name in temp_file_map.
    """
    global temp_file_map
    file_info = temp_file_map.get(unique_id)

    if file_info and os.path.exists(file_info['file_path']):
        file_path = file_info['file_path']
        download_name = file_info['download_name']
        temp_dir_for_cleanup = file_info.get('temp_dir') # Get the temp dir associated with this file
        
        print(f"DEBUG: Serving file: {file_path} as {download_name}")

        @after_this_request
        def remove_file(response):
            """
            This function will be called after the response has been sent.
            It cleans up the temporary file and its directory.
            """
            if unique_id in temp_file_map:
                del temp_file_map[unique_id] # Remove from map
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path) # Delete the file
                    print(f"DEBUG: Deleted file: {file_path}")
                except OSError as e:
                    print(f"WARNING: Could not delete file {file_path} via after_this_request: {e}")
            
            # Attempt to remove the parent temporary directory if it's empty
            if temp_dir_for_cleanup and os.path.exists(temp_dir_for_cleanup):
                try:
                    # Check if the directory is empty before trying to remove
                    if not os.listdir(temp_dir_for_cleanup):
                        shutil.rmtree(temp_dir_for_cleanup)
                        print(f"DEBUG: Deleted empty temporary directory: {temp_dir_for_cleanup}")
                    else:
                        print(f"DEBUG: Temporary directory {temp_dir_for_cleanup} is not empty, skipping removal.")
                except OSError as e:
                    print(f"WARNING: Could not remove temporary directory {temp_dir_for_cleanup} via after_this_request: {e}")
            return response

        return send_file(file_path, as_attachment=True, download_name=download_name, conditional=True)
    else:
        print(f"ERROR: File info not found for unique_id: {unique_id} or file does not exist at expected path.")
        return jsonify({"error": "File not found or already deleted."}), 404

def extract_interview_params_llm(text):
    """
    Extracts job_role, difficulty, num_questions, and job_description from text using LLM.
    """
    schema = {
        "type": "object",
        "properties": {
            "job_role": {"type": "string", "description": "The job role mentioned, e.g., 'Software Engineer'."},
            "difficulty": {"type": "string", "enum": ["easy", "normal", "hard", "difficult"], "description": "The desired difficulty level."},
            "num_questions": {"type": "integer", "description": "The number of questions requested."},
            "job_description": {"type": "string", "description": "Any provided job description or context."}
        },
        "required": ["job_role", "difficulty", "num_questions"], # Make these required for a valid request
        "propertyOrdering": ["job_role", "difficulty", "num_questions", "job_description"]
    }

    prompt = f"""
    Extract the following parameters from the user's request:
    - job_role (e.g., 'Software Engineer', 'Data Scientist', 'Project Manager')
    - difficulty (choose from 'easy', 'normal', 'hard', 'difficult')
    - num_questions (an integer, default to 5 if not specified)
    - job_description (any additional context about the job, optional)

    If a parameter is not explicitly mentioned, infer a reasonable default or leave it null.
    For difficulty, if a general request is made, default to 'normal'.
    For job_role, if not specified, infer from context or use a generic term like 'general'.

    Respond ONLY with a JSON object matching the provided schema.

    User request: "{text}"
    """
    
    try:
        extracted_data = orchestrate_llm_calls(prompt, response_schema=schema)
        if extracted_data:
            # Provide defaults if LLM misses something, or normalize
            job_role = extracted_data.get('job_role', 'general').lower().strip()
            difficulty = extracted_data.get('difficulty', 'normal').lower().strip()
            num_questions = int(extracted_data.get('num_questions', 5)) # Ensure it's an int
            job_description = extracted_data.get('job_description', '').strip()

            # Basic normalization for job roles to match dataset if possible
            if 'software' in job_role or 'dev' in job_role: job_role = 'software engineer'
            elif 'data scien' in job_role: job_role = 'data scientist'
            elif 'project' in job_role or 'pm' in job_role: job_role = 'project manager'
            elif 'ux' in job_role or 'user experience' in job_role: job_role = 'ux designer'
            elif 'marketing' in job_role: job_role = 'marketing specialist'
            elif 'financial' in job_role or 'finance' in job_role: job_role = 'financial analyst'
            
            # Ensure difficulty is one of the allowed enums
            if difficulty not in ["easy", "normal", "hard", "difficult"]:
                difficulty = "normal" # Default if invalid

            return job_role, difficulty, num_questions, job_description
        return None, None, None, None
    except Exception as e:
        print(f"ERROR: LLM extraction of interview parameters failed: {e}")
        return None, None, None, None

def handle_interview_qa(job_role, difficulty, num_questions, job_description):
    """
    Generates interview questions, prioritizing local data, then falling back to LLM.
    """
    questions = []
    local_fallback_message = "I couldn't find specific local questions for your request."

    # 1. Try local lookup first
    if INTERVIEW_QA_LOADED:
        try:
            # Normalize inputs for lookup
            job_role_lower = job_role.lower()
            difficulty_lower = difficulty.lower()

            # Filter by job role (using contains for flexibility)
            filtered_by_role = interview_qa_df[
                interview_qa_df['job_role'].str.contains(job_role_lower, case=False, na=False)
            ]
            
            # Filter by difficulty
            filtered_by_difficulty = filtered_by_role[
                filtered_by_role['difficulty'] == difficulty_lower
            ]

            if not filtered_by_difficulty.empty:
                # Get unique questions to avoid duplicates if any
                available_questions = filtered_by_difficulty['question'].tolist()
                
                # Randomly select up to num_questions
                import random
                if len(available_questions) > num_questions:
                    questions.extend(random.sample(available_questions, num_questions))
                else:
                    questions.extend(available_questions)
                    local_fallback_message = f"Found {len(available_questions)} local questions. "
                
                if questions:
                    print(f"DEBUG: Found {len(questions)} local interview questions for {job_role} ({difficulty}).")
                    local_fallback_response = "Here are some local questions I found:\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
                    return local_fallback_response # Return local response if successful
                else:
                    local_fallback_response = f"No specific local questions found for '{job_role}' at '{difficulty}' difficulty."

        except Exception as e:
            print(f"ERROR: Error during local interview Q&A lookup: {e}")
            local_fallback_response = f"Local interview Q&A lookup failed: {e}. Falling back to LLM APIs."
    else:
        local_fallback_response = "Local interview Q&A dataset not loaded. Falling back to LLM APIs."

    # 2. Fallback to LLM if local questions are insufficient or not found
    print(f"DEBUG: Local questions insufficient or failed. Calling LLM for interview questions.")
    llm_prompt = f"""
    Generate {num_questions} interview questions for a {difficulty} {job_role} role.
    
    If a job description is provided, use it to tailor the questions:
    Job Description: "{job_description if job_description else 'N/A'}"

    Focus on {difficulty} level questions.
    Provide only the questions, numbered, without any introductory or concluding remarks.
    """
    
    try:
        llm_response = orchestrate_llm_calls(llm_prompt)
        if llm_response and llm_response.strip():
            # Attempt to parse LLM's numbered list
            llm_questions = re.findall(r'^\d+\.\s*(.*)', llm_response, re.MULTILINE)
            if llm_questions:
                # Add LLM questions, ensuring we don't exceed num_questions in total
                for q in llm_questions:
                    if len(questions) < num_questions: # Only add if we still need more questions
                        questions.append(q.strip())
                    else:
                        break
                print(f"DEBUG: Supplemented with {len(llm_questions)} questions from LLM.")
            else:
                print("WARNING: LLM response for interview questions could not be parsed as a numbered list.")
                if not questions: # If no local questions were found AND LLM parsing failed
                    questions.append(llm_response.strip())
        else:
            print("WARNING: LLM returned empty or unsatisfactory response for interview questions.")
    except Exception as e:
        print(f"ERROR: Error calling LLM orchestration for interview Q&A: {e}")



    # 3. Final response formatting
    if questions:
        final_response_text = f"Here are {len(questions)} interview questions for a **{job_role}** role at **{difficulty}** difficulty:\n\n"
        for i, q in enumerate(questions):
            final_response_text += f"{i+1}. {q}\n"
    else:
        final_response_text = local_fallback_response # This will be the message from local lookup if both fail

    return final_response_text

def extract_translation_request(text):
    """
    Extracts the text to translate, translation command, and target language from user input.
    Expected format: "text to translate" "translate" "target language"
    or: "text to translate" "translate this" "target language"
    """
    # Regex to capture: (text before first quote) "command" "language"
    # Group 1: text_to_translate
    # Group 2: command (e.g., "translate", "translate this")
    # Group 3: target_language
    match = re.match(r'^(.*?)\s*"(translate|translate this)"\s*"(.*?)"$', text.strip(), re.IGNORECASE)
    if match:
        text_to_translate = match.group(1).strip()
        command = match.group(2).lower().strip()
        target_language = match.group(3).strip().lower() # Ensure target language is lowercase
        print(f"DEBUG: Extracted for translation: Text='{text_to_translate[:50]}...', Command='{command}', Target='{target_language}'")
        return text_to_translate, command, target_language
    print(f"DEBUG: Could not extract translation request from: {text}")
    return None, None, None

def extract_summarization_request(text):
    """
    Extracts the content (text or URL) to summarize, summarization command, and optional word limit.
    Expected formats:
    - "long text content" "Summarise"
    - "long text content" "Summarise" "100 Words"
    - "[https://example.com/article](https://example.com/article)" "Summaries the article"
    - "[https://example.com/article](https://example.com/article)" "Article Summarise" "under 200 words"
    """
    # Regex to capture: (content_to_summarize) "command" (optional "word_limit")
    # Group 1: content_to_summarize (text or URL)
    # Group 2: command (e.g., "summarise", "summaries the article", "article summarise")
    # Group 3: optional_word_limit (e.g., "under 100 Words", "100 Words")
    match = re.match(r'^(.*?)\s*"(summarise|summaries the article|article summarise)"(?:\s*"(.*?)")?$', text.strip(), re.IGNORECASE)
    
    if match:
        content_to_summarize = match.group(1).strip()
        command = match.group(2).lower().strip()
        word_limit_str = match.group(3)
        
        summary_word_limit = None
        if word_limit_str:
            num_match = re.search(r'(\d+)\s*words?', word_limit_str, re.IGNORECASE)
            if num_match:
                summary_word_limit = int(num_match.group(1))
        
        print(f"DEBUG: Extracted for summarization: Content='{content_to_summarize[:50]}...', Command='{command}', WordLimit={summary_word_limit}")
        return content_to_summarize, command, summary_word_limit
    
    print(f"DEBUG: Could not extract summarization request from: {text}")
    return None, None, None

def extract_tts_request(text):
    """
    Extracts the text/URL for TTS, the TTS command, and the optional voice role.
    Expected formats:
    - "text to speak" "Audio"
    - "text to speak" "Text To Audio" "Male"
    - "[https://example.com/article](https://example.com/article)" "Speech" "Female"
    - "[https://example.com/article](https://example.com/article)" "Text to Speech"
    """
    # Regex to capture: (content_for_tts) "command" (optional "voice_role")
    # Group 1: content_for_tts (text or URL)
    # Group 2: command (e.g., "audio", "text to audio", "speech", "text to speech")
    # Group 3: optional_voice_role (e.g., "Male", "Female", "Child")
    match = re.match(r'^(.*?)\s*"(audio|text to audio|speech|text to speech)"(?:\s*"(.*?)")?$', text.strip(), re.IGNORECASE)
    
    if match:
        content_for_tts = match.group(1).strip()
        command = match.group(2).lower().strip()
        voice_role_str = match.group(3)
        
        voice_role = voice_role_str.lower().strip() if voice_role_str else "default"
        
        print(f"DEBUG: Extracted for TTS: Content='{content_for_tts[:50]}...', Command='{command}', VoiceRole='{voice_role}'")
        return content_for_tts, command, voice_role
    
    print(f"DEBUG: Could not extract TTS request from: {text}")
    return None, None, None


# Extracts a recipe query and command from user input.
def extract_recipe_request(text, keywords):
    """
    Extracts the recipe query and command from user input.
    Example: 'pasta, tomato, garlic "recipe"' -> ('pasta, tomato, garlic', 'recipe')
    """
    match = re.match(r'^(.*?)\s*"([^"]+)"\s*$', text.strip())
    if match:
        query = match.group(1).strip()
        command = match.group(2).strip().lower()
        if any(kw in command for kw in keywords):
            return query, command
    return None, None


# Extracts a paraphrasing request from user input.
def extract_paraphrasing_request(text, paraphrasing_keywords):
    """
    Checks if paraphrasing keyword is present and extracts original input.
    Expected format: 'Some text here.' "rewrite"
    """
    if any(kw in text.lower() for kw in paraphrasing_keywords):
        match = re.match(r'^(.*?)\s*"(rewrite|paraphrase|paraphrase the text|rewriter the text)"$', text.strip(), re.IGNORECASE)
        if match:
            input_text = match.group(1).strip()
            command = match.group(2).strip().lower()
            return input_text, command
    return None, None

# Extracts an API test request from user input.
def extract_api_test_request(text, api_test_keywords):
    """
    Extracts API URL, optional JSON body, and command from user message.
    Expected: <URL> <optional json> "test api"
    """
    if any(kw in text.lower() for kw in api_test_keywords):
        try:
            match = re.match(r'^(https?://\S+)\s*(\{.*\})?\s*"(test api|api test|test|api)"$', text.strip(), re.IGNORECASE)
            if match:
                url = match.group(1)
                json_body = match.group(2)
                command = match.group(3)
                payload = json.loads(json_body) if json_body else None
                return url, payload, command
        except Exception as e:
            print(f"Error extracting API test request: {e}")
    return None, None, None

# Extracts a grammar request from user input.
def extract_grammar_request(message):
    try:
        # Handle file uploads (you can integrate better file handling if needed)
        if message.strip().startswith("file:"):
            path = message.replace("file:", "").strip()
            if path.endswith(".txt"):
                with open(path, 'r') as f:
                    return f.read()
            elif path.endswith(".pdf"):
                import PyPDF2
                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif path.endswith(".docx"):
                import docx
                doc = docx.Document(path)
                return " ".join([para.text for para in doc.paragraphs])
        # Handle link
        elif message.strip().startswith("http"):
            import requests
            from bs4 import BeautifulSoup
            res = requests.get(message.strip(), timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            return soup.get_text()
        else:
            return message
    except Exception as e:
        print("Error extracting grammar input:", e)
        return message



def map_voice_role_to_gemini_voice(role):
    """
    Maps a user-friendly voice role to a specific Gemini prebuilt voice name.
    """
    role_map = {
        "male": "kore",
        "female": "erinome",
        "child": "leda",
        "matured man": "gacrux",
        "alpha male": "orus", # Stronger, more authoritative
        "alpha female": "alnilam", # Firm, clear
        "mature female": "gacrux", # Gacrux is mature, can work for both
        "boy": "leda",
        "younger boy": "leda",
        "girl": "autonoe",
        "younger girl": "autonoe",
        "soft man": "iapetus",
        "soft women": "vindemiatrix",
        "rude men": "algenib", # Gravelly, can imply rudeness with prompt tone
        "rude women": "charon", # Informative, but can be prompted for rude tone
        "default": "kore" # Default if no specific role is matched
    }
    return role_map.get(role.lower(), "kore") # Default to Kore if not found

def pcm_to_wav(pcm_data, sample_rate, num_channels=1, sample_width=2):
    """
    Converts raw PCM audio data (bytes) to a WAV file format (bytes).
    Assumes 16-bit signed PCM.
    """
    # Decode base64 PCM data
    decoded_pcm_data = base64.b64decode(pcm_data)
    
    # Convert bytes to numpy array of int16
    audio_array = np.frombuffer(decoded_pcm_data, dtype=np.int16)

    # Create an in-memory WAV file
    with io.BytesIO() as buffer:
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width) # 2 bytes for 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_array.tobytes()) # Write the raw bytes from numpy array
        return buffer.getvalue()

def fetch_article_content(url):
    """
    Fetches the main textual content, title, and URL from a given article URL.
    Returns a dictionary with 'title', 'content', 'url'.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        soup = BeautifulSoup(response.text, 'lxml') # Use lxml for faster parsing

        # Extract title
        title = soup.find('title').get_text(strip=True) if soup.find('title') else "No Title Found"

        # Try to find common article content containers
        article_content_tags = ['article', 'main', 'div', 'p']
        main_content = []
        
        for tag_name in article_content_tags:
            for tag in soup.find_all(tag_name):
                # Heuristic: prioritize tags with more text or specific classes
                text = tag.get_text(separator=' ', strip=True)
                if len(text.split()) > 50: # Consider it main content if it has substantial text
                    main_content.append(text)
                elif tag_name == 'article' or (tag.has_attr('class') and any(cls in str(tag['class']).lower() for cls in ['article', 'content', 'body', 'post'])):
                    main_content.append(text)
        
        content = " ".join(main_content)
        
        # Fallback: if no structured content, just take all text from body
        if not content.strip():
            content = soup.body.get_text(separator=' ', strip=True) if soup.body else ""

        # Clean up extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()

        return {
            'title': title,
            'content': content,
            'url': url
        }

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch article from {url}: {e}")
        return {'title': 'Error', 'content': f"Could not fetch article from {url}. Reason: {e}", 'url': url}
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during article fetching for {url}: {e}")
        return {'title': 'Error', 'content': f"An unexpected error occurred while processing the article from {url}. Reason: {e}", 'url': url}

def local_summarize_text(text, word_limit=None):
    """
    Performs basic extractive summarization using NLTK.
    """
    if not NLTK_AVAILABLE:
        return None
    
    try:
        sentences = sent_tokenize(text)
        # Calculate word frequencies
        word_frequencies = {}
        stop_words = set(stopwords.words('english'))
        for word in word_tokenize(text.lower()):
            if word.isalnum() and word not in stop_words:
                word_frequencies[word] = word_frequencies.get(word, 0) + 1
        
        # Rank sentences by sum of word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    sentence_scores[i] = sentence_scores.get(i, 0) + word_frequencies[word]
        
        # Sort sentences by score in descending order
        ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        summary_sentences = []
        current_word_count = 0
        
        # Reconstruct summary in original sentence order
        original_order_indices = sorted([idx for idx, _ in ranked_sentences])
        
        for idx in original_order_indices:
            if idx < len(sentences): # Ensure index is valid
                sentence = sentences[idx]
                words_in_sentence = len(word_tokenize(sentence))
                if word_limit is None or (current_word_count + words_in_sentence <= word_limit):
                    summary_sentences.append(sentence)
                    current_word_count += words_in_sentence
                elif current_word_count < word_limit: # Add partial sentence if it fits
                    remaining_words = word_limit - current_word_count
                    truncated_sentence = " ".join(word_tokenize(sentence)[:remaining_words])
                    if truncated_sentence:
                        summary_sentences.append(truncated_sentence + "...")
                        current_word_count += len(word_tokenize(truncated_sentence))
                    break # Stop after adding partial sentence
                else:
                    break # Stop if word limit is reached
        
        summary = " ".join(summary_sentences)
        print(f"DEBUG: Local summarization successful (approx. {len(word_tokenize(summary))} words).")
        return summary
    except Exception as e:
        print(f"ERROR: Local summarization failed: {e}")
        return None

def handle_summarization(content_input, summary_word_limit):
    """
    Summarizes text or article content, handling word limits and providing details.
    Prioritizes local summarization, then falls back to LLM.
    """
    is_url = False
    original_url = None
    article_title = None
    original_content_for_display = content_input
    truncated_flag = False

    # 1. Determine if input is URL or raw text
    if re.match(r'https?://[^\s]+', content_input):
        is_url = True
        original_url = content_input
        print(f"DEBUG: Summarization request is a URL: {original_url}")
        article_data = fetch_article_content(original_url)
        
        if "Error" in article_data['title']:
            return f"I couldn't summarize the article from {original_url}. {article_data['content']}"
        
        article_title = article_data['title']
        original_content_for_summarization = article_data['content']
        original_content_for_display = article_data['content'] # For displaying the full original if not truncated
    else:
        original_content_for_summarization = content_input
        print(f"DEBUG: Summarization request is raw text.")

    # Apply backend input word limit
    words = original_content_for_summarization.split()
    if len(words) > SUMMARIZATION_INPUT_WORD_LIMIT:
        original_content_for_summarization = " ".join(words[:SUMMARIZATION_INPUT_WORD_LIMIT])
        truncated_flag = True
        print(f"DEBUG: Summarization input text truncated. Original word count: {len(words)}, Truncated to: {SUMMARIZATION_INPUT_WORD_LIMIT}")
    
    summarized_text = None
    local_fallback_message = "Local summarization failed or was unavailable. Attempting API summarization."

    # 2. Try local summarization first
    if NLTK_AVAILABLE:
        print("DEBUG: Attempting local summarization...")
        try:
            summarized_text = local_summarize_text(original_content_for_summarization, summary_word_limit)
            if summarized_text:
                print("DEBUG: Local summarization successful.")
            else:
                print("DEBUG: Local summarization returned empty. Falling back to API.")
        except Exception as e:
            print(f"ERROR: Local summarization failed: {e}. Falling back to API.")
            local_fallback_message = f"Local summarization failed: {e}. Attempting API summarization."
    else:
        print("DEBUG: NLTK not available. Falling back to API summarization.")
        local_fallback_message = "NLTK not available for local summarization. Attempting API summarization."

    # 3. Fallback to LLM summarization if local fails or is not available
    if not summarized_text:
        prompt = f"Summarize the following text."
        if summary_word_limit:
            prompt += f" The summary should be approximately {summary_word_limit} words long."
        else:
            prompt += " Make it concise and informative."
        
        prompt += f"\n\nText to summarize:\n```\n{original_content_for_summarization}\n```"

        print("DEBUG: Calling LLM for summarization...")
        summarized_text = orchestrate_llm_calls(prompt)
        if not summarized_text:
            summarized_text = local_fallback_message + "\n\nI apologize, I could not generate a summary for the provided content using any available method."


    if summarized_text and summarized_text.strip():
        response_parts = []
        if is_url:
            response_parts.append(f"**Original Article:** [{article_title}]({original_url})\n")
        
        if truncated_flag:
            response_parts.append(f"*(Note: The input text was limited to {SUMMARIZATION_INPUT_WORD_LIMIT} words for summarization.)*\n")
            response_parts.append(f"**Original Text (first {SUMMARIZATION_INPUT_WORD_LIMIT} words):**\n```\n{original_content_for_summarization}\n```\n")
        else:
            if not is_url: # Only show raw original text if it wasn't a URL and not truncated
                response_parts.append(f"**Original Text:**\n```\n{original_content_for_display}\n```\n")

        response_parts.append(f"**Summary:**\n```\n{summarized_text}\n```")
        return "\n".join(response_parts)
    else:
        return "I apologize, I could not generate a summary for the provided content. The LLM might have encountered an issue or the content might be too complex."

def local_text_to_speech(text, voice_role):
    """
    Converts text to speech using pyttsx3.
    Returns the path to the temporary WAV file, or None on failure.
    """
    if not PYTTSX3_AVAILABLE:
        return None

    try:
        engine = pyttsx3.init()
        
        # Attempt to set voice based on role
        voices = engine.getProperty('voices')
        selected_voice = None
        
        # Simple mapping for pyttsx3 voices (platform dependent)
        if "male" in voice_role:
            for voice in voices:
                if voice.gender == 'male' or 'male' in voice.name.lower():
                    selected_voice = voice.id
                    break
        elif "female" in voice_role:
            for voice in voices:
                if voice.gender == 'female' or 'female' in voice.name.lower():
                    selected_voice = voice.id
                    break
        elif "child" in voice_role or "boy" in voice_role or "girl" in voice_role or "younger" in voice_role:
            for voice in voices:
                if 'child' in voice.name.lower() or 'young' in voice.name.lower():
                    selected_voice = voice.id
                    break
        
        if selected_voice:
            engine.setProperty('voice', selected_voice)
            print(f"DEBUG: pyttsx3 using voice: {engine.getProperty('voice')}")
        else:
            print("DEBUG: No specific pyttsx3 voice found for role, using default.")

        temp_dir = tempfile.mkdtemp()
        audio_filename = f"local_tts_output_{os.urandom(8).hex()}.wav"
        audio_file_path = os.path.join(temp_dir, audio_filename)
        
        engine.save_to_file(text, audio_file_path)
        engine.runAndWait()
        
        if os.path.exists(audio_file_path):
            print(f"DEBUG: Local TTS audio saved to: {audio_file_path}")
            return audio_file_path, audio_filename, temp_dir
        else:
            print("ERROR: pyttsx3 failed to save audio file.")
            return None, None, None
    except Exception as e:
        print(f"ERROR: Local Text-to-Speech (pyttsx3) failed: {e}")
        return None, None, None

# Generate QR code locally using qrcode library
def generate_qr_code_local(data):
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color='black', back_color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"ERROR: Local QR code generation failed: {e}")
        return None

# Generate barcode locally using python-barcode library
def generate_barcode_local(data):
    try:
        CODE128 = barcode.get_barcode_class('code128')
        code128 = CODE128(data, writer=ImageWriter())
        buf = io.BytesIO()
        code128.write(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"ERROR: Local barcode generation failed: {e}")
        return None


# Decode QR code or barcode from image bytes using pyzbar
def decode_qr_or_barcode_local(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        decoded_objs = pyzbar_decode(img)
        results = []
        for obj in decoded_objs:
            results.append({'type': obj.type, 'data': obj.data.decode('utf-8')})
        return results
    except Exception as e:
        print(f"ERROR: Local QR/barcode decoding failed: {e}")
        return None


# Save a temporary image buffer to a file and return a download link

def save_temp_image_and_get_link(img_buf, prefix="qr", ext="png", delete_after_sec=120):
    """
    Saves an image buffer to a temp file and schedules deletion.
    Returns (file_path, download_url, filename).
    """
    temp_dir = tempfile.mkdtemp()
    filename = f"{prefix}_{os.urandom(8).hex()}.{ext}"
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "wb") as f:
        f.write(img_buf.getvalue())
    unique_id = os.urandom(16).hex()
    temp_file_map[unique_id] = {
        'file_path': file_path,
        'download_name': filename,
        'temp_dir': temp_dir
    }
    # Schedule deletion after delete_after_sec seconds
    def delayed_delete():
        time.sleep(delete_after_sec)
        if unique_id in temp_file_map:
            try:
                os.remove(file_path)
                shutil.rmtree(temp_dir)
                del temp_file_map[unique_id]
                print(f"DEBUG: Deleted temp QR/barcode file and dir: {file_path}")
            except Exception as e:
                print(f"WARNING: Could not delete temp QR/barcode file: {e}")
    threading.Thread(target=delayed_delete, daemon=True).start()
    return file_path, f"/download_media/{unique_id}", filename






# Local plagiarism check using difflib

def local_plagiarism_check(user_text, plagiarism_corpus_df, similarity_threshold=0.85):
    """
    Checks the user_text against the local plagiarism corpus DataFrame.
    Returns a list of dicts with matched_text and similarity score.
    """
    import difflib
    matches = []
    user_text_lower = user_text.lower()
    for idx, row in plagiarism_corpus_df.iterrows():
        corpus_text = str(row['phrase']).lower()
        # Use SequenceMatcher to get similarity ratio
        similarity = difflib.SequenceMatcher(None, user_text_lower, corpus_text).ratio()
        if similarity >= similarity_threshold:
            matches.append({
                "matched_text": row['phrase'],
                "similarity": similarity
            })
    return matches

# Local code generation lookup
def local_code_generation(user_prompt, code_generation_df):
    """
    Looks up the local code generation dataset for a matching prompt.
    Returns the code snippet if found, else None.
    """
    # Simple contains match (can be improved with embeddings or fuzzy matching)
    matches = code_generation_df[code_generation_df['prompt'].str.lower().str.contains(user_prompt.lower())]
    if not matches.empty:
        # Return the first match (or random.choice for variety)
        row = matches.iloc[0]
        return f"**Language:** {row['language']}\n```{row['language'].lower()}\n{row['code']}\n```"
    return None

# Local recipe lookup
def local_recipe_lookup(query, recipe_df):
    """
    Looks up recipes by name or ingredients in the local dataset.
    Returns a list of matching recipes (dicts).
    """
    query_lower = query.lower()
    # Try by recipe name
    matches = recipe_df[recipe_df['recipe_name'].str.lower().str.contains(query_lower, na=False)]
    # If not found, try by ingredients
    if matches.empty:
        for ingredient in query_lower.split(','):
            ingredient = ingredient.strip()
            if ingredient:
                matches = recipe_df[recipe_df['ingredients'].str.lower().str.contains(ingredient, na=False)]
                if not matches.empty:
                    break
    return matches.to_dict(orient='records') if not matches.empty else []

# Handle text-to-speech requests, prioritizing local TTS, then falling back to Gemini TTS API
def handle_text_to_speech(content_input, voice_role):
    """
    Converts text or article content to speech, saves it as a WAV, and provides a download link.
    Prioritizes local TTS (pyttsx3), then falls back to Gemini TTS API.
    """
    is_url = False
    original_url = None
    article_title = None
    original_text_for_display = content_input
    truncated_flag = False
    
    # 1. Determine if input is URL or raw text
    if re.match(r'https?://[^\s]+', content_input):
        is_url = True
        original_url = content_input
        print(f"DEBUG: TTS request is a URL: {original_url}")
        article_data = fetch_article_content(original_url)
        
        if "Error" in article_data['title']:
            return {'type': 'error', 'content': f"I couldn't fetch the article from {original_url} for TTS. {article_data['content']}"}
        
        article_title = article_data['title']
        text_for_tts = article_data['content']
        original_text_for_display = article_data['content'] # For displaying the full original if not truncated
    else:
        text_for_tts = content_input
        print(f"DEBUG: TTS request is raw text.")

    # Apply backend input word limit
    words = text_for_tts.split()
    if len(words) > TTS_INPUT_WORD_LIMIT:
        text_for_tts = " ".join(words[:TTS_INPUT_WORD_LIMIT])
        truncated_flag = True
        print(f"DEBUG: TTS input text truncated. Original word count: {len(words)}, Truncated to: {TTS_INPUT_WORD_LIMIT}")
    
    if not text_for_tts.strip():
        return {'type': 'error', 'content': "No text provided or extracted for Text-to-Speech."}

    audio_file_path = None
    audio_filename = None
    temp_dir = None
    
    # 2. Try local TTS (pyttsx3) first
    if PYTTSX3_AVAILABLE:
        print("DEBUG: Attempting local Text-to-Speech (pyttsx3)...")
        try:
            audio_file_path, audio_filename, temp_dir = local_text_to_speech(text_for_tts, voice_role)
            if audio_file_path:
                print("DEBUG: Local TTS successful.")
        except Exception as e:
            print(f"ERROR: Local TTS failed: {e}. Falling back to API.")

    # 3. Fallback to LLM (Gemini TTS API) if local fails or is not available
    if not audio_file_path:
        print("DEBUG: Local TTS failed or not available. Calling LLM for TTS...")
        gemini_voice_name = map_voice_role_to_gemini_voice(voice_role)
        audio_config = {
            "voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": gemini_voice_name}
            }
        }
        tts_response_data = orchestrate_llm_calls(text_for_tts, audio_config=audio_config)

        if tts_response_data and 'audio_data' in tts_response_data:
            try:
                pcm_audio_data = tts_response_data['audio_data']
                sample_rate = tts_response_data['sample_rate']
                
                # Convert PCM to WAV
                wav_file_bytes = pcm_to_wav(pcm_audio_data, sample_rate)
                
                # Save the WAV file to a temporary location
                temp_dir = tempfile.mkdtemp()
                audio_filename = f"api_tts_output_{os.urandom(8).hex()}.wav"
                audio_file_path = os.path.join(temp_dir, audio_filename)
                
                with open(audio_file_path, 'wb') as f:
                    f.write(wav_file_bytes)
                
                print(f"DEBUG: API TTS audio saved to: {audio_file_path}")
            except Exception as e:
                print(f"ERROR: Error during API TTS audio processing or saving: {e}")
                return {'type': 'error', 'content': f"An error occurred while processing the audio via API: {e}"}
        else:
            return {'type': 'error', 'content': "I apologize, I could not generate speech for the provided text using any available method. The TTS service might be unavailable or encountered an error."}

    response_content = f"**Original Text:**\n```\n{original_text_for_display}\n```\n\n"
    if is_url:
        response_content = f"**Original Article:** [{article_title}]({original_url})\n\n" + response_content
    if truncated_flag:
        response_content += f"*(Note: Text was limited to {TTS_INPUT_WORD_LIMIT} words for speech generation.)*\n"
    
    response_content += f"Here is the audio for your text (Voice: {voice_role.capitalize()}):"
    
    return {
        'type': 'download_link', # Reusing download_link type for consistency
        'file_path': audio_file_path,
        'filename': audio_filename,
        'temp_dir': temp_dir,
        'response_text': response_content # Custom text to display in chat
    }



def api_plagiarism_check(user_text):
    """
    Checks for plagiarism using LLM APIs as a fallback.
    Returns a dictionary with the result.
    """
    prompt = (
        "Check the following text for plagiarism. "
        "If any part is plagiarized, highlight the plagiarized segments using **:red[text]** markdown. "
        "Respond with 'No plagiarism detected.' if the text is original.\n\n"
        f"Text:\n\"\"\"\n{user_text}\n\"\"\""
    )
    try:
        response = orchestrate_llm_calls(prompt)
        if response:
            return {"plagiarism": response}
        else:
            return {"plagiarism": "Could not check plagiarism via API."}
    except Exception as e:
        print(f"API plagiarism check failed: {e}")
        return {"plagiarism": f"API error: {e}"}


# Handles API test requests by first trying a local call, then falling back to LLM for explanation.
def handle_api_test_request(url, payload=None):
    """
    Performs local test of API first (via requests).
    If fails, falls back to API (LLM) for explanation.
    """
    import time
    headers = {'Content-Type': 'application/json'}
    method = "POST" if payload else "GET"
    start_time = time.time()

    # Step 1: Local API Call
    try:
        if method == "POST":
            response = requests.post(url, json=payload, headers=headers, timeout=10)
        else:
            response = requests.get(url, headers=headers, timeout=10)

        end_time = time.time()
        response_text = response.text[:1000]  # Trim very long responses

        return {
            "method": method,
            "url": url,
            "status_code": response.status_code,
            "response": response_text,
            "elapsed_time": round(end_time - start_time, 2),
            "used": "local"
        }
    except Exception as e:
        print(f"Local API call failed: {e}")

        # Step 2: Use LLM to interpret (fallback)
        try:
            prompt = f"""
You are an API tester. Help the user test this API call and show sample output.

URL: {url}
Payload: {json.dumps(payload) if payload else "None"}
Method: {"POST" if payload else "GET"}

Return a friendly explanation or formatted output.
            """
            response = orchestrate_llm_calls(prompt)
            return {
                "method": method,
                "url": url,
                "status_code": None,
                "response": response,
                "elapsed_time": None,
                "used": "api"
            }
        except Exception as ex:
            print(f"LLM fallback also failed: {ex}")
            return {
                "method": method,
                "url": url,
                "status_code": None,
                "response": f"Error testing the API: {ex}",
                "elapsed_time": None,
                "used": "error"
            }



# Attempts paraphrasing using LLM API first, then falls back to local NLTK-based logic.
def handle_paraphrasing_request(text_to_rewrite):
    """
    Attempts paraphrasing using LLM API first, then falls back to NLTK-based local logic.
    Returns: rewritten_text, method_used ("api" or "local")
    """
    prompt = f"Rewrite the following text for clarity and originality:\n\n\"{text_to_rewrite}\""
    
    # Step 1: Try API-based paraphrasing
    try:
        rewritten = orchestrate_llm_calls(prompt)
        if rewritten:
            return rewritten.strip(), "api"
    except Exception as e:
        print(f"LLM paraphrasing failed: {e}")

    # Step 2: Local NLP-based fallback
    try:
        from nltk.corpus import wordnet
        from nltk.tokenize import word_tokenize
        words = word_tokenize(text_to_rewrite)
        new_words = []
        for word in words:
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name().replace("_", " ")
                new_words.append(synonym)
            else:
                new_words.append(word)
        local_rewritten = " ".join(new_words)
        return local_rewritten, "local"
    except Exception as e:
        print(f"Local paraphrasing failed: {e}")
        return text_to_rewrite, "original"


# Handles grammar correction requests, prioritizing local tools and falling back to API if needed.
def handle_grammar_request(text):
    from spellchecker import SpellChecker
    from textblob import TextBlob

    final_result = {"corrected": "", "source": "local"}

    # ✅ Step 0: Truncate long text
    if len(text) > 2000:
        text = text[:2000]

    prompt = f"Correct the grammar and spelling in the following paragraph:\n\n{text}"

    # ✅ Step 1: Try AI model first via orchestrator
    try:
        ai_response = orchestrate_llm_calls(prompt, preferred="gemini")
        final_result["corrected"] = ai_response
        final_result["source"] = "api"
        return final_result
    except Exception as api_error:
        print("AI grammar model failed, fallback to local. Error:", api_error)

    # ✅ Step 2: Local fallback (SpellChecker + TextBlob)
    try:
        corrected_text = text

        # 🔍 Spell check
        spell = SpellChecker()
        words = text.split()
        misspelled = spell.unknown(words)
        for word in misspelled:
            correction = spell.correction(word)
            if correction:
                corrected_text = corrected_text.replace(word, correction)

        # ✍️ Grammar fix
        blob = TextBlob(corrected_text)
        corrected_text = str(blob.correct())

        final_result["corrected"] = corrected_text
        final_result["source"] = "local"

    except Exception as local_error:
        print("Local grammar correction failed:", local_error)
        final_result["corrected"] = text  # return original text as fallback

    return final_result







# Uses LLM APIs to generate QR codes or barcodes from data.
def generate_qr_code_api(data):
    try:
        url = f"https://api.qrserver.com/v1/create-qr-code/?data={data}&size=200x200"
        response = requests.get(url)
        if response.status_code == 200:
            return io.BytesIO(response.content)
        else:
            return None
    except Exception as e:
        print(f"ERROR: API QR code generation failed: {e}")
        return None


# Uses LLM APIs to generate code from a natural language prompt.
def api_code_generation(user_prompt):
    """
    Uses LLM APIs to generate code from a natural language prompt.
    """
    prompt = (
        f"Generate code for the following request. Respond ONLY with the code block, "
        f"and specify the language if possible.\n\nRequest: {user_prompt}"
    )
    try:
        response = orchestrate_llm_calls(prompt)
        if response:
            return response
        else:
            return "Could not generate code via API."
    except Exception as e:
        print(f"API code generation failed: {e}")
        return f"API error: {e}"

# Uses LLM APIs to look up recipes based on a query.
def api_recipe_lookup(query):
    """
    Uses LLM APIs to generate a recipe for the given query.
    Returns a dict with recipe, image_url, video_url, pros, cons, nutrient_values, and calories if possible.
    """
    prompt = (
        f"Suggest a recipe based on the following input (ingredients or recipe name): '{query}'. "
        "Provide the recipe name, a list of ingredients, step-by-step instructions, "
        "pros, cons, nutrient values (as a string), and calories (as a string), "
        "and if possible, a relevant image URL and a YouTube video link for making the recipe. "
        "Respond in JSON with keys: recipe_name, ingredients, instructions, pros, cons, nutrient_values, calories, image_url, video_url."
    )
    try:
        response = orchestrate_llm_calls(prompt)
        try:
            recipe = json.loads(response)
            return [recipe]
        except Exception:
            return [{
                "recipe_name": "Recipe",
                "ingredients": query,
                "instructions": response,
                "pros": "",
                "cons": "",
                "nutrient_values": "",
                "calories": "",
                "image_url": "",
                "video_url": ""
            }]
    except Exception as e:
        print(f"API recipe lookup failed: {e}")
        return []

# Paraphrases the given text for clarity.
def paraphrase_text(text):
    prompt = f"Paraphrase the following text for clarity:\n\n{text}"
    try:
        ai_response = orchestrate_llm_calls(prompt, preferred="gemini")
        return {"paraphrased": ai_response, "source": "ai"}
    except Exception as e:
        return {"paraphrased": text, "source": "fallback"}





@app.route('/chat', methods=['POST'])

def chat():
    """
    Handles chat requests with multi-language support, prioritizing local models,
    and automatically detecting location from user text. It also orchestrates
    LLM API calls with fallback logic and handles media download/extraction tasks.
    Now includes AI-Based Interview Q&A, Text Translation, and Text-to-Speech.
    """
    user_message = request.form.get('message', '').strip()
    
    if not user_message:
        return jsonify({'response': "I'm sorry, I didn't receive a message."})

    ai_response_english = ""
    local_fallback_response = ""
    media_result = None # Initialize media_result here to ensure it's always defined

    try:
        # --- Define all keyword lists at the beginning of the function ---
        # These lists MUST be defined here to be in scope for all subsequent conditional checks.
        code_keywords = ["correct", "correct the code", "analyse the code", "analyze code", "detect code language", "fix this code", "fix it", "code language", "code correction", "code analysis", "debug", "error in code", "code error", "code debugging", "code detect", "code detection", "code analyse", "code analyze", "code diagnose", "code diagnosis", "code diagnose analysis", "code diagnosis analysis", "code diagnose detect", "code diagnosis detect", "code diagnose detection", "code diagnosis detection", "code diagnose analyse", "code diagnosis analyse", "code diagnose analyze", "code diagnosis analyze", "code diagnose prediction", "code diagnosis prediction", "code diagnose prediction analysis", "code diagnosis prediction analysis", "code diagnose prediction detect", "code diagnosis prediction detect", "code diagnose prediction detection", "code diagnosis prediction detection", "code diagnose prediction analyse", "code diagnosis prediction analyse", "code diagnose prediction analyze", "code diagnosis prediction analyze", "code diagnose prediction diagnose", "code diagnosis prediction diagnose", "code diagnose prediction diagnose analysis", "code diagnosis prediction diagnose analysis", "code diagnose prediction diagnose detect", "code diagnosis prediction diagnose detect", "code diagnose prediction diagnose detection", "code diagnosis prediction diagnose detection", "code diagnose prediction diagnose analyse", "code diagnosis prediction diagnose analyse", "code diagnose prediction diagnose analyze", "code diagnosis prediction diagnose analyze", "code diagnose prediction diagnose prediction", "code diagnosis prediction diagnose prediction", "code diagnose prediction diagnose prediction analysis", "code diagnosis prediction diagnose prediction analysis", "code diagnose prediction diagnose prediction detect", "code diagnosis prediction diagnose prediction detect", "code diagnose prediction diagnose prediction detection", "code diagnosis prediction diagnose prediction detection", "code diagnose prediction diagnose prediction analyse", "code diagnosis prediction diagnose prediction analyse", "code diagnose prediction diagnose prediction analyze", "code diagnosis prediction diagnose prediction analyze", "code diagnose prediction detect", "code diagnosis prediction analyze"]
        
        disease_keywords = ["detect disease", "disease", "predict disease", "symptoms", "disease prediction", "disease detection", "disease symptoms", "predict disease symptoms", "disease analysis", "disease diagnose", "disease diagnosis", "disease prediction symptoms", "disease diagnose symptoms", "disease diagnose analysis", "disease diagnosis analysis", "disease diagnose prediction", "disease diagnosis prediction", "disease diagnose prediction symptoms", "disease diagnosis prediction symptoms", "disease diagnose prediction analysis", "disease diagnosis prediction analysis", "disease diagnose prediction detect", "disease diagnosis prediction detect", "disease diagnose prediction detection", "disease diagnosis prediction detection", "disease diagnose prediction analyse", "disease diagnosis prediction analyse", "disease diagnose prediction analyze", "disease diagnosis prediction analyze", "disease diagnose prediction diagnose", "disease diagnosis prediction diagnose", "disease diagnose prediction diagnose analysis", "disease diagnosis prediction diagnose analysis", "disease diagnose prediction diagnose detect", "disease diagnosis prediction diagnose detect", "disease diagnose prediction diagnose detection", "disease diagnosis prediction diagnose detection", "disease diagnose prediction diagnose analyse", "disease diagnosis prediction diagnose analyse", "disease diagnose prediction diagnose analyze", "disease diagnosis prediction diagnose analyze", "disease diagnose prediction diagnose prediction", "disease diagnosis prediction diagnose prediction", "disease diagnose prediction diagnose prediction analysis", "disease diagnosis prediction diagnose prediction analysis", "disease diagnose prediction diagnose prediction detect", "disease diagnosis prediction diagnose prediction detect", "disease diagnose prediction diagnose prediction detection", "disease diagnosis prediction diagnose prediction detection", "disease diagnose prediction diagnose prediction analyse", "disease diagnosis prediction diagnose prediction analyse", "disease diagnose prediction diagnose prediction analyze", "disease diagnosis prediction diagnose prediction analyze"]
        
        study_keywords = ["study", "qualification", "how to study", "guid", "guide", "process", "university", "apply job", "job process", "job qualification", "job study", "job guide", "job guidance", "job qualification process", "job study process", "job guide process", "job guidance process", "job qualification guide", "job study guide", "job guide qualification", "job guidance qualification", "job qualification study", "job study qualification", "job guide study", "job guidance study", "job qualification process guide", "job study process guide", "job guide process guide", "job guidance process guide", "job qualification process study", "job study process study", "job guide process study", "job guidance process study", "job qualification guide study", "job study guide study", "job guide qualification study", "job guidance qualification study", "job qualification study guide", "job study qualification guide", "job guide study guide", "job guidance study guide", "job qualification process guide study", "job study process guide study", "job guide process guide study", "job guidance process guide study", "job qualification process study guide", "job study process study guide", "job guide process study guide", "job guidance process study guide", "job qualification guide study guide", "job study guide study guide", "job guide qualification study guide", "job guidance qualification study guide", "job qualification study guide study", "job study qualification guide study", "job guide study qualification", "job guidance study qualification", "job qualification process guide qualification", "job study process guide qualification", "job guide process guide qualification", "job guidance process guide qualification", "job qualification process study qualification", "job study process study qualification", "job guide process study qualification", "job guidance process study qualification", "job qualification guide study qualification", "job study guide study qualification", "job guide qualification study qualification", "job guidance qualification study qualification", "job qualification study guide qualification", "job study qualification guide qualification", "job guide study qualification qualification", "job guidance study qualification qualification", "job qualification process guide qualification study", "job study process guide qualification study", "job guide process guide qualification study", "job guidance process guide qualification study", "job qualification process study qualification study", "job study process study qualification study", "job guide process study qualification study", "job guidance process study qualification study", "job qualification guide study qualification study", "job study guide study qualification study", "job guide qualification study qualification study", "job guidance qualification study qualification study", "job qualification study guide qualification study", "job study qualification guide qualification study", "job guide study qualification qualification study", "job guidance study qualification qualification study"]
        
        sentiment_keywords = ["sentiment", "analyse sentiment", "analyze sentiment", "sentiment analysis", "sentiment detect", "sentiment detection", "sentiment score", "sentiment score analysis", "sentiment score detect", "sentiment score detection", "sentiment score analyse", "sentiment score analyze", "sentiment score prediction", "sentiment score predict", "sentiment score prediction analysis", "sentiment score prediction detect", "sentiment score prediction detection", "sentiment score prediction analyse", "sentiment score prediction analyze", "sentiment score prediction diagnose", "sentiment score prediction diagnosis", "sentiment score prediction diagnose analysis", "sentiment score prediction diagnosis analysis", "sentiment score prediction diagnose detect", "sentiment score prediction diagnosis detect", "sentiment score prediction diagnose detection", "sentiment score prediction diagnosis detection", "sentiment score prediction diagnose analyse", "sentiment score prediction diagnosis analyse", "sentiment score prediction diagnose analyze", "sentiment score prediction diagnosis analyze", "sentiment score prediction diagnose prediction", "sentiment score prediction diagnosis prediction", "sentiment score prediction diagnose prediction analysis", "sentiment score prediction diagnosis prediction analysis", "sentiment score prediction diagnose prediction detect", "sentiment score prediction diagnosis prediction detect", "sentiment score prediction diagnose prediction detection", "sentiment score prediction diagnosis prediction detection", "sentiment score prediction diagnose prediction analyse", "sentiment score prediction diagnosis prediction analyse", "sentiment score prediction diagnose prediction analyze", "sentiment score prediction diagnosis prediction analyze", "sentiment score prediction diagnose prediction diagnose", "sentiment score prediction diagnosis prediction diagnose", "sentiment score prediction diagnose prediction diagnose analysis", "sentiment score prediction diagnosis prediction diagnose analysis", "sentiment score prediction diagnose prediction diagnose detect", "sentiment score prediction diagnosis prediction diagnose detect", "sentiment score prediction diagnose prediction diagnose detection", "sentiment score prediction diagnosis prediction diagnose detection", "sentiment score prediction diagnose prediction diagnose analyse", "sentiment score prediction diagnosis prediction diagnose analyse", "sentiment score prediction diagnose prediction diagnose analyze", "sentiment score prediction diagnosis prediction diagnose analyze"]

        fake_news_keywords = ["fake", "news", "is fake", "is news", "fack check", "real or fake", "verify news", "verify article", "verify news article", "verify news content", "verify article content", "verify news text", "verify article text", "verify news input", "verify article input", "verify news input text", "verify article input text", "verify news input content", "verify article input content", "verify news input article", "verify article input article", "verify news input article content", "verify article input article content", "verify news input article text", "verify article input article text", "verify news input article text content", "verify article input article text content", "verify news input article content text", "verify article input article content text", "verify news input article text content input", "verify article input article text content input", "verify news input article content text input", "verify article input article content text input", "verify news input article text content input text", "verify article input article text content input text"]

        job_recommendation_keywords = ["jobe", "preferred job", "job recommendation", "job suggestions", "job advice", "career recommendation", "career advice", "job search", "find job", "job opportunities", "job openings", "job vacancies", "job leads", "job prospects", "career opportunities", "career openings", "career vacancies", "career leads", "career prospects", "job matching", "career matching", "job fit", "career fit", "job compatibility", "career compatibility", "job alignment", "career alignment", "job suitability", "career suitability", "job search strategy", "career search strategy", "job search tips", "career search tips", "job search advice", "career search advice", "job search guidance", "career search guidance", "job search resources", "career search resources", "job search tools", "career search tools", "job search platforms", "career search platforms", "job search websites", "career search websites", "job search engines", "career search engines", "job search apps", "career search apps", "job search software", "career search software", "job search services", "career search services", "job search agencies", "career search agencies", "job search consultants", "career search consultants", "job search coaches", "career search coaches", "job search mentors", "career search mentors", "job search networks", "career search networks", "job search communities", "career search communities"]

        spam_keywords = ["spam", "junk", "unsolicited", "advertisement", "buy now", "click here", "subscribe", "free offer", "winner", "claim prize", "limited time", "urgent", "act now", "risk-free", "guaranteed", "no cost", "exclusive deal", "special promotion", "unsecured credit", "debt relief", "make money fast", "work from home", "earn cash", "get paid to click", "online business opportunity"] 

        # Keywords for media handling
        media_task_keywords = ["download", "download video", "video download", "audio download", "download thumbnail", "extract video script", "extract video tags", "extract video views and earnings", "extract video description", "extract video title", "extract video content","extract video metadata", "extract audio", "extract video", "extract video content", "extract video metadata", "extract video transcript", "extract video subtitles" "extract video captions", "extract video comments", "extract video likes","extract video shares", "extract video statistics", "extract video information", "extract video details", "extract video data", "extract video insights", "extract video summary", "extract video highlights", "extract video key points"]
        
        # Keywords for Interview Q&A
        interview_keywords = ["interview", "interview q&a", "q&a", "job q&a", "job interview", "interview questions", "interview questions and answers", "job interview questions", "job interview q&a", "job interview questions and answers", "interview questions for", "interview questions for job", "job interview questions for", "interview questions for job role", "interview questions for job role", "interview questions for job position", "job interview questions for job position"]

        # Keywords for Text Translation
        translation_keywords = ["translate", "translate this" "translate the text", "translate the article", "translate the content", "translate the document", "translate the input", "translate the input text", "translate the input article", "translate the input document", "translate the input content", "text to translate", "translate this text", "translate this article", "translate this content", "translate this document"]

        # Keywords for Text Summarization
        summarization_keywords = ["summarise", "summaries the article", "article summarise", "summarize", "summarize this", "summarize the text", "summarize the article", "summarize the content", "summarize the document", "summarize the text", "summarize the input", "summarize the input text", "summarize the input article", "summarize the input document", "summarize the input content"]

        # Keywords for Text-to-Speech
        tts_keywords = ["audio", "text to audio", "speech", "text to speech", "text to speech", "text to speak", "speak this", "read this out loud", "read this", "read this text", "read this article", "read this content", "read this document", "read this input", "read this input text", "read this input article", "read this input document", "read this input content", "text to read", "text to read out loud", "text to read this", "text to read this out loud", "text to read this text", "text to read this article", "text to read this content", "text to read this document", "text to read this input", "text to read this input text", "text to read this input article", "text to read this input document", "text to read this input content", "read this text out loud", "read this article out loud", "read this content out loud", "read this document out loud", "read this input out loud", "read this input text out loud", "read this input article out loud", "read this input document out loud", "read this input content out loud", "text to read out loud this", "text to read out loud this text", "text to read out loud this article", "text to read out loud this content", "text to read out loud this document", "text to read out loud this input", "text to read out loud this input text", "text to read out loud this input article", "text to read out loud this input document", "text to read out loud this input content", "read this text out loud this", "read this article out loud this", "read this content out loud this", "read this document out loud this", "read this input out loud this", "read this input text out loud this", "read this input article out loud this", "read this input document out loud this", "read this input content out loud this"]

        # Voice Role Keywords
        voice_role_keywords = ["male", "female", "child", "matured man", "alpha male", "alpha female", "mature female", "boy", "younger boy", "girl", "younger girl", "soft man", "soft women", "rude men", "rude women", "default", "standard", "normal", "robotic", "neutral", "calm", "energetic", "excited", "sad", "angry", "happy", "joyful", "serious", "funny", "friendly", "professional", "casual", "authoritative", "confident", "sympathetic", "empathetic", "persuasive", "motivational", "inspirational", "narrative", "storytelling", "dramatic", "intense", "soothing", "relaxing", "uplifting", "encouraging", "supportive", "reassuring", "informative", "educational", "conversational", "engaging", "dynamic", "expressive", "articulate", "clear", "precise", "detailed", "thorough", "comprehensive", "insightful", "thought-provoking", "analytical", "critical", "objective", "subjective", "balanced", "unbiased", "fair-minded", "open-minded", "curious", "inquisitive", "exploratory", "adventurous", "bold", "courageous", "fearless", "daring", "risk-taking", "innovative", "creative", "imaginative", "visionary", "futuristic", "progressive", "forward-thinking", "revolutionary", "groundbreaking", "pioneering", "trailblazing"]

        #Plagiarism detection keywords
        plagiarism_keywords = ["plagiarism", "check plagiarism", "plagiarism check", "detect plagiarism", "plagiarism detection", "is this plagiarised", "is this original", "originality check", "is this content original", "is this content plagiarised", "plagiarism free", "plagiarism checker", "detect plagiarism", "plagiarism", "plagiarism detect", "detect is an ai writed", "detect is non human writed", "detect is an human writed"]
        # Keywords for Code Generation
        code_generation_keywords = ["generate code", "code", "code generate", "code generation", "create code", "write code", "produce code", "make code", "code for", "code snippet" "code example", "code sample", "code script", "code function", "code module", "code library", "code class", "code method", "code algorithm", "code logic", "code solution", "code implementation" "code design", "code architecture", "code pattern", "code structure", "code framework", "code template", "code boilerplate", "code snippet generation", "generate code snippet", "generate code example", "generate code sample", "generate code script", "generate code function", "generate code module", "generate code library", "generate code class", "generate code method", "generate code algorithm", "generate code logic", "generate code solution", "generate code implementation" , "generate code design", "generate code architecture", "generate code pattern", "generate code structure", "generate code framework", "generate code template" , "generate code boilerplate", "generate code snippet generation"]
        # Keywords for QR Code Generation
        qr_keywords = ["qr", "generate qr code", "generate qr", "qr code", "qr generate", "qr generator", "content into qr code", "content qr code" "content into qr", "content qr" "content into qr code", "content qr code", "qr code generate", "qr code generator", "qr generate code", "qr generate content", "qr content generate", "qr content generator", "qr code content generate", "qr code content generator" "qr code content into", "qr code content into generate", "qr code content into generator", "qr code content into qr", "qr code content into qr generate", "qr code content into qr generator" "qr code content into qr code", "qr code content into qr code generate", "qr code content into qr code generator"]
        # Keywords for Barcode Generation
        barcode_keywords = ["barcode", "bar code", "barcode generate", "generate barcode", "barcode generator", "bar code generate", "generate bar code", "bar code generator", "content into barcode", "content barcode", "content into bar code", "content bar code", "content into barcode generate", "content barcode generate", "content into bar code generate", "content bar code generate", "barcode content generate", "barcode content generator", "bar code content generate", "bar code content generator", "barcode code content generate", "barcode code content generator", "barcode code content into", "barcode code content into generate", "barcode code content into generator", "barcode code content into barcode", "barcode code content into barcode generate", "barcode code content into barcode generator", "barcode code content into bar code", "barcode code content into bar code generator", "barcode code content into bar code generate"]
        # Keywords for Recipe Suggestions
        recipe_keywords = ["recipe", "how to cook", "how to make this recipe", "how to make", "cook", "cooking", "make recipe", "suggest recipe","recipe suggestions", "recipe ideas", "recipe suggestion", "recipe idea", "recipe suggestions", "recipe ideas", "recipe suggestion", "recipe idea", "cooking recipe", "cooking recipes", "recipe cooking", "recipe cookings", "cooking recipe suggestions", "cooking recipe ideas", "cooking recipe suggestion", "cooking recipe idea", "recipe cooking suggestions", "recipe cooking ideas", "recipe cooking suggestion", "recipe cooking idea"]
        # --- Paraphrasing Keywords ---
        paraphrasing_keywords = ["rewrite", "paraphrase", "paraphrase the text", "rewriter the text", "rephrase", "rephrase the text", "rephrase the article", "rephrase the content", "rephrase the document", "rephrase the input", "rephrase the input text", "rephrase the input article", "rephrase the input document", "rephrase the input content", "rewrite the text", "rewrite the article", "rewrite the content", "rewrite the document", "rewrite the input", "rewrite the input text", "rewrite the input article", "rewrite the input document",  "rephrase this", "rewrite this", "paraphrase this", "paraphrase the article", "paraphrase the content", "paraphrase the document", "paraphrase the input", "paraphrase the input text", "paraphrase the input article", "paraphrase the input document", "paraphrase the input content", "rewrite the input content"]
        # Keywords for API Testing
        api_test_keywords = ["test api", "api test", "test", "api", "api testing", "test api endpoint", "api endpoint test", "test api functionality", "api functionality test", "test api response", "api response test", "test api performance", "api performance test", "test api security", "api security test", "test api reliability", "api reliability test", "test api usability", "api usability test", "test api compatibility", "api compatibility test", "test api integration", "api integration test", "test api load", "api load test", "test api stress", "api stress test", "test api scalability", "api scalability test", "test api availability", "api availability test", "test api functionality and performance", "api functionality and performance test", "test api functionality and security", "api functionality and security test", "test api functionality and reliability", "api functionality and reliability test", "test api functionality and usability", "api functionality and usability test", "test api functionality and compatibility", "api functionality and compatibility test", "test api functionality and integration", "api functionality and integration test", "test api functionality and load", "api functionality and load test", "test api functionality and stress", "api functionality and stress test", "test api functionality and scalability", "api functionality and scalability test", "test api functionality and availability", "api functionality and availability test", "test api performance and security", "api performance and security test", "test api performance and reliability", "api performance and reliability test", "test api performance and usability", "api performance and usability test", "test api performance and compatibility", "api performance and compatibility test", "test api performance and integration", "api performance and integration test", "test api performance and load", "api performance and load test", "test api performance and stress", "api performance and stress test", "test api performance and scalability", "api performance and scalability test", "test api performance and availability", "api performance and availability test"]
        # Keywords for Grammar Correction
        grammar_keywords = ["spell", "spell check", "check spell", "grammar", "check grammar", "spelling", "grammar check", "correct grammar", "correct spell", "grammar correction", "spell correction", "grammar fix", "spell fix", "grammar error", "spell error", "grammar mistake", "spell mistake", "grammar issue", "spell issue", "grammar problem", "spell problem", "grammar proofread", "spell proofread", "grammar proofreading", "spell proofreading", "grammar edit", "spell edit", "grammar editing", "spell editing", "grammar correct", "spell correct", "grammar correction check", "spell correction check", "grammar fix check", "spell fix check", "grammar error check", "spell error check", "grammar mistake check", "spell mistake check", "grammar issue check", "spell issue check", "grammar problem check", "spell problem check", "grammar proofread check", "spell proofread check", "grammar proofreading check", "spell proofreading check", "grammar edit check", "spell edit check", "grammar editing check", "spell editing check", "grammar correct check", "spell correct check", "grammar correction fix", "spell correction fix", "grammar fix fix", "spell fix fix", "grammar error fix", "spell error fix", "grammar mistake fix", "spell mistake fix", "grammar issue fix", "spell issue fix", "grammar problem fix", "spell problem fix", "grammar proofread fix", "spell proofread fix", "grammar proofreading fix", "spell proofreading fix", "grammar edit fix", "spell edit fix", "grammar editing fix", "spell editing fix", "grammar correct fix", "spell correct fix", "grammar correction error", "spell correction error", "grammar fix error", "spell fix error", "grammar error error", "spell error error", "grammar mistake error", "spell mistake error", "grammar issue error", "spell issue error", "grammar problem error", "spell problem error", "grammar proofread error", "spell proofread error", "grammar proofreading error", "spell proofreading error", "grammar edit error", "spell edit error", "grammar editing error", "spell editing error", "grammar correct error", "spell correct error"]
        # --- Priority: Code Analysis ---

        print("DEBUG: Keyword lists defined and in scope for the current chat request.")
        
        # Step 1: Detect User Language
        print(f"DEBUG: Original user message: '{user_message}'")
        detected_lang = detect_user_language(user_message)
        print(f"DEBUG: Detected language: '{detected_lang}'")
        processed_message = user_message # Message to be used for local/LLM processing

        # Step 2: Translate to English if not English
        if detected_lang != 'en':
            print(f"DEBUG: User message detected as '{detected_lang}'. Translating to English for processing...")
            try:
                processed_message = translate_text_llm(user_message, 'en', detected_lang)
                if not processed_message:
                    raise ValueError("Translation to English resulted in empty text.")
                print(f"DEBUG: Translated to English: {processed_message}")
            except Exception as e:
                print(f"ERROR: Error translating to English: {e}. Proceeding with original message.")
                processed_message = user_message
        else:
            print("DEBUG: User language is English. No translation to English needed for processing.")


        # --- Priority: Plagiarism Detection ---
        if any(keyword in processed_message.lower() for keyword in plagiarism_keywords):
            local_fallback_response = "I couldn't perform local plagiarism detection for your request."
            plagiarized_segments = []
            if PLAGIARISM_CORPUS_LOADED:
                try:
                    plagiarized_segments = local_plagiarism_check(processed_message, plagiarism_corpus_df)
                    if plagiarized_segments:
                        highlighted = processed_message
                        for seg in plagiarized_segments:
                            highlighted = highlighted.replace(seg["matched_text"], f"**:red[{seg['matched_text']}]**")
                        ai_response_english = (
                            f"Plagiarism detected! Highlighted below:\n\n{highlighted}\n\n"
                            f"Similarity scores: {[round(seg['similarity'], 2) for seg in plagiarized_segments]}"
                        )
                    else:
                        ai_response_english = "No plagiarism detected in your text (local corpus)."
                except Exception as e:
                    print(f"ERROR: Local plagiarism check failed: {e}")
                    local_fallback_response = f"Local plagiarism check failed: {e}. Falling back to API."
            else:
                print("DEBUG: Local plagiarism corpus not loaded. Falling back to API.")
                local_fallback_response = "Local plagiarism corpus not loaded. Falling back to API."

            # API fallback
            if not plagiarized_segments:
                api_result = api_plagiarism_check(processed_message)
                if api_result and "plagiarism" in api_result:
                    ai_response_english = f"API Plagiarism Check: {api_result['plagiarism']}"
                else:
                    ai_response_english = local_fallback_response

            return jsonify({'response': ai_response_english})
        
        # --- Priority: Code Analysis ---

        # Step 3: Automatically Extract Location from Processed Message (if applicable)
        user_location = extract_location_from_text(processed_message)
        if user_location:
            print(f"DEBUG: Detected user location from text: {user_location}")
        else:
            print("DEBUG: No specific location detected in user message.")

        # Determine if the user's query is a "quoted task" (general quoted task, not specific to translation/summarization/TTS)
        is_general_quoted_task = False
        general_quoted_content_match = re.match(r'^"([^"]+)"$', user_message.strip())
        if general_quoted_content_match:
            is_general_quoted_task = True
            # For general quoted tasks, the processed_message is the content inside quotes
            # However, for translation/summarization/TTS, we need the *full* original user_message for parsing.
            # So, we'll use user_message for these specific parsing tasks, and processed_message for other tasks.
            print(f"DEBUG: Detected general quoted task. Processed message (for non-specific tasks): '{processed_message}'")
        
        # --- Prioritized Processing Logic ---
        # Local models are attempted first, then fallback to LLM orchestration.
        # If a general quoted task is detected, it forces LLM orchestration only if local model fails.

        # --- Priority 0: Media Download/Extraction ---
        url, command = extract_url_and_command(user_message) # Use original user_message for URL extraction
        if url and command and any(kw in command for kw in media_task_keywords):
            print(f"DEBUG: Detected media task: URL='{url}', Command='{command}'")
            media_result = handle_media_task(url, command) # Assign media_result here
            
            if media_result['type'] == 'download_link':
                # Store the full path and the desired download name in the global map
                unique_id = os.urandom(16).hex()
                temp_file_map[unique_id] = {
                    'file_path': media_result['file_path'],
                    'download_name': media_result['filename'],
                    'temp_dir': media_result['temp_dir'] # Store temp_dir for later cleanup
                }
                
                download_relative_url = f"/download_media/{unique_id}"
                ai_response_english = f"Your requested file '{media_result['filename']}' is ready."
                
                # Return this special structure to Streamlit
                return jsonify({
                    'response': ai_response_english,
                    'download_info': {
                        'download_url': download_relative_url,
                        'filename': media_result['filename'],
                        'type': 'media_download' # Indicate it's a media download
                    }
                })
            elif media_result['type'] == 'text':
                ai_response_english = media_result['content']
                if detected_lang != 'en':
                    final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                    return jsonify({'response': final_response})
                else:
                    return jsonify({'response': ai_response_english})
            elif media_result['type'] == 'error':
                local_fallback_response = media_result['content']
                # If local media handling failed, try LLM for a more general explanation
                prompt_for_llm_fallback = f"I tried to process the request: '{user_message}', but encountered an issue: {local_fallback_response}. Can you provide a general explanation or alternative suggestion for this type of task?"
                ai_response_english = orchestrate_llm_calls(prompt_for_llm_fallback)
                if not ai_response_english: # If LLM also failed or returned empty
                    ai_response_english = local_fallback_response
                
                if detected_lang != 'en':
                    final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                    return jsonify({'response': final_response})
                else:
                    return jsonify({'response': ai_response_english})

        
            # --- Paraphrasing Feature ---
            text_to_rewrite, paraphrase_cmd = extract_paraphrasing_request(user_message)
            if text_to_rewrite and paraphrase_cmd:
                rewritten_text, method = handle_paraphrasing_request(text_to_rewrite)
                return jsonify({
                    "response": f"**Original Text:** {text_to_rewrite}\n\n**{paraphrase_cmd.title()} ({method.upper()}):** {rewritten_text}"
                })


        # --- Priority: Spell , gramma  checking---
        if any(keyword in user_message.lower() for keyword in grammar_keywords):
            extracted_text = extract_grammar_request(user_message)
            result = handle_grammar_request(extracted_text)
            response = f"""**Grammar & Spell Check Result ({result['source'].upper()}):**

                **Original:**  
                {extracted_text}

                **Corrected:**  
                {result['corrected']}
                """
            return jsonify({"response": response})



        # --- Priority: Recipe Generator ---
        recipe_query, recipe_command = extract_recipe_request(user_message, recipe_keywords)
        if recipe_query and recipe_command:
            print(f"DEBUG: Detected recipe generation task. Query: {recipe_query}")
            recipes = []
            local_fallback_response = "I couldn't find a local recipe for your request."
            # Try local dataset first
            if RECIPE_DATASET_LOADED:
                try:
                    recipes = local_recipe_lookup(recipe_query, recipe_df)
                except Exception as e:
                    print(f"ERROR: Local recipe lookup failed: {e}")
                    local_fallback_response = f"Local recipe lookup failed: {e}. Falling back to API."
            else:
                print("DEBUG: Local recipe dataset not loaded. Falling back to API.")
                local_fallback_response = "Local recipe dataset not loaded. Falling back to API."

            # API fallback
            if not recipes:
                recipes = api_recipe_lookup(recipe_query)

            # Format response for Streamlit
            if recipes:
                recipe = recipes[0]  # Show the first match

                # Format ingredients as a bulleted list
                ingredients = recipe.get('ingredients', '')
                if isinstance(ingredients, str):
                    ingredients_list = [i.strip() for i in re.split(r',|\n', ingredients) if i.strip()]
                else:
                    ingredients_list = ingredients

                # Format instructions as a numbered list
                instructions = recipe.get('instructions', '')
                if isinstance(instructions, str):
                    steps = [s.strip() for s in re.split(r'\n|(?<=\.)\s+', instructions) if s.strip()]
                else:
                    steps = instructions

                response_parts = [
                    f"# 🍽️ {recipe.get('recipe_name', 'Recipe')}\n"
                ]
                if ingredients_list:
                    response_parts.append("**📝 Ingredients:**")
                    response_parts.append("\n".join([f"- {i}" for i in ingredients_list]))
                if steps:
                    response_parts.append("\n**👨‍🍳 Instructions:**")
                    response_parts.append("\n".join([f"{idx+1}. {step}" for idx, step in enumerate(steps)]))
                if recipe.get('pros'):
                    response_parts.append(f"\n**✅ Pros:** {recipe['pros']}")
                if recipe.get('cons'):
                    response_parts.append(f"\n**⚠️ Cons:** {recipe['cons']}")
                if recipe.get('nutrient_values'):
                    response_parts.append(f"\n**🥗 Nutrient Values:** {recipe['nutrient_values']}")
                if recipe.get('calories'):
                    response_parts.append(f"\n**🔥 Calories:** {recipe['calories']}")
                if recipe.get('image_url'):
                    response_parts.append(f"\n![Recipe Image]({recipe['image_url']})")
                if recipe.get('video_url'):
                    response_parts.append(f"\n▶️ [Watch Video]({recipe['video_url']})")

                ai_response_english = "\n\n".join(response_parts)
            else:
                ai_response_english = local_fallback_response

            # ...after ai_response_english is built...
            image_url = recipe.get('image_url', '')
            video_url = recipe.get('video_url', '')
            print("DEBUG Flask recipe image_url:", image_url)
            return jsonify({
                'response': ai_response_english,
                'recipe_image_url': image_url,
                'recipe_video_url': video_url
            })



        # --- API Testing Feature ---
        api_url, payload, command = extract_api_test_request(user_message, api_test_keywords)
        if api_url and command:
            result = handle_api_test_request(api_url, payload)
            response = f"""**API Tester Result ({result['used'].upper()}):**

                - **Method:** {result['method']}
                - **URL:** `{result['url']}`
                - **Status Code:** {result['status_code'] if result['status_code'] else "N/A"}
                - **Response:**
                ```json
                {result['response']}
            ```"""

            if result['elapsed_time']:
                response += f"\n- **Time Taken:** {result['elapsed_time']} seconds"

            return jsonify({"response": response})

        
            

        # --- Priority 1: Text-to-Speech (NEW FEATURE) ---
        content_for_tts_raw, tts_command, voice_role = extract_tts_request(user_message)
        
        if content_for_tts_raw and tts_command and any(kw in tts_command for kw in tts_keywords):
            print(f"DEBUG: Detected Text-to-Speech task. Voice Role: {voice_role}")
            tts_result = handle_text_to_speech(content_for_tts_raw, voice_role)
            
            if tts_result['type'] == 'download_link': # Reusing download_link type
                unique_id = os.urandom(16).hex()
                temp_file_map[unique_id] = {
                    'file_path': tts_result['file_path'],
                    'download_name': tts_result['filename'],
                    'temp_dir': tts_result['temp_dir']
                }
                download_relative_url = f"/download_media/{unique_id}"
                
                return jsonify({
                    'response': tts_result['response_text'], # Custom text for TTS
                    'download_info': {
                        'download_url': download_relative_url,
                        'filename': tts_result['filename'],
                        'type': 'audio_tts' # Indicate it's an audio TTS file
                    }
                })
            elif tts_result['type'] == 'error':
                ai_response_english = tts_result['content']
                if detected_lang != 'en':
                    final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                    return jsonify({'response': final_response})
                else:
                    return jsonify({'response': ai_response_english})



        # --- Priority: QR/Barcode Generation/Decoding ---
        # --- Priority: QR/Barcode Generation/Decoding ---
        data_for_code, code_command = extract_data_and_command(user_message, qr_keywords + barcode_keywords)
        if data_for_code and code_command:
            if any(kw in code_command for kw in qr_keywords):
                buf = generate_qr_code_local(data_for_code)
                if not buf:
                    buf = generate_qr_code_api(data_for_code)
                if buf:
                    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    file_path, download_url, filename = save_temp_image_and_get_link(buf, prefix="qr", ext="png")
                    return jsonify({
                        'response': "QR code generated.",
                        'qr_image_base64': img_b64,
                        'download_info': {
                            'download_url': download_url,
                            'filename': filename,
                            'type': 'qr_code'
                        }
                    })
                else:
                    return jsonify({'response': "Failed to generate QR code."})

            elif any(kw in code_command for kw in barcode_keywords):
                buf = generate_barcode_local(data_for_code)
                if buf:
                    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    file_path, download_url, filename = save_temp_image_and_get_link(buf, prefix="barcode", ext="png")
                    return jsonify({
                        'response': "Barcode generated.",
                        'barcode_image_base64': img_b64,
                        'download_info': {
                            'download_url': download_url,
                            'filename': filename,
                            'type': 'barcode'
                        }
                    })
                else:
                    return jsonify({'response': "Failed to generate barcode."})





        # --- Priority: Code Generation ---
        code_gen_prompt, code_gen_command, _ = extract_translation_request(user_message)  # Reuse extraction logic for quoted commands
        if code_gen_prompt and code_gen_command and any(kw in code_gen_command.lower() for kw in code_generation_keywords):
            print(f"DEBUG: Detected code generation task. Prompt: {code_gen_prompt}")
            local_fallback_response = "I couldn't generate code for your request locally."
            code_snippet = None

            # Try local dataset first
            if CODE_GENERATION_LOADED:
                try:
                    code_snippet = local_code_generation(code_gen_prompt, code_generation_df)
                    if code_snippet:
                        ai_response_english = code_snippet
                    else:
                        ai_response_english = local_fallback_response
                except Exception as e:
                    print(f"ERROR: Local code generation failed: {e}")
                    ai_response_english = f"Local code generation failed: {e}. Falling back to API."
            else:
                print("DEBUG: Local code generation dataset not loaded. Falling back to API.")
                ai_response_english = "Local code generation dataset not loaded. Falling back to API."

            # API fallback
            if not code_snippet or "couldn't" in ai_response_english.lower():
                api_result = api_code_generation(code_gen_prompt)
                ai_response_english = api_result if api_result else ai_response_english

            return jsonify({'response': ai_response_english})




        # --- Priority 2: Text Translation ---
        text_to_translate_raw, translation_command, target_language = extract_translation_request(user_message)
        
        if text_to_translate_raw and translation_command and target_language and any(kw in translation_command for kw in translation_keywords):
            print(f"DEBUG: Detected text translation task. Target Language: {target_language}")
            original_text_for_display = text_to_translate_raw
            
            # Apply word limit
            words = original_text_for_display.split()
            truncated_flag = False
            if len(words) > TRANSLATION_WORD_LIMIT:
                original_text_for_translation = " ".join(words[:TRANSLATION_WORD_LIMIT])
                truncated_flag = True
                print(f"DEBUG: Text truncated for translation. Original word count: {len(words)}, Truncated to: {TRANSLATION_WORD_LIMIT}")
            else:
                original_text_for_translation = original_text_for_display
            
            # Detect source language for better translation accuracy
            source_lang = detect_user_language(original_text_for_translation)
            
            translated_text = translate_text_llm(original_text_for_translation, target_language, source_lang)

            if translated_text:
                ai_response_english = f"**Original Text ({source_lang.upper()}):**\n```\n{original_text_for_display}\n```\n\n"
                ai_response_english += f"**Translated Text ({target_language.upper()}):**\n```\n{translated_text}\n```\n"
                if truncated_flag:
                    ai_response_english += f"\n*(Note: Your original text was limited to {TRANSLATION_WORD_LIMIT} words for translation.)*"
            else:
                ai_response_english = f"I apologize, I could not translate the text to {target_language}. Please ensure the target language is valid and try again."
            
            return jsonify({'response': ai_response_english})

        # --- Priority 3: Text Summarization ---
        content_to_summarize, summarization_command, summary_word_limit = extract_summarization_request(user_message)
        
        if content_to_summarize and summarization_command and any(kw in summarization_command for kw in summarization_keywords):
            print(f"DEBUG: Detected text summarization task. Summary Word Limit: {summary_word_limit}")
            ai_response_english = handle_summarization(content_to_summarize, summary_word_limit)
            
            return jsonify({'response': ai_response_english})


        # --- Priority 4: Interview Q&A ---
        elif any(keyword in processed_message.lower() for keyword in interview_keywords):
            print(f"DEBUG: Detected interview Q&A task.")
            job_role, difficulty, num_questions, job_description = extract_interview_params_llm(processed_message)

            if job_role and difficulty and num_questions:
                ai_response_english = handle_interview_qa(job_role, difficulty, num_questions, job_description)
            else:
                ai_response_english = "I couldn't understand the job role, difficulty, or number of questions for the interview Q&A. Please specify them clearly (e.g., 'Interview for Software Engineer, hard difficulty, 5 questions')."
            
            if detected_lang != 'en':
                final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                return jsonify({'response': final_response})
            else:
                return jsonify({'response': ai_response_english})


        # --- Priority 5: Fake News Detection ---
        elif any(keyword in processed_message.lower() for keyword in fake_news_keywords):
            local_fallback_response = "I couldn't perform local fake news detection for your request."
            
            content_to_analyze, original_query_text, content_type = extract_url_or_content_for_fake_news(processed_message)

            if FAKE_NEWS_LOADED and content_type == "text" and content_to_analyze and not is_general_quoted_task:
                try:
                    text_for_analysis = content_to_analyze
                    for keyword in fake_news_keywords:
                        text_for_analysis = text_for_analysis.replace(keyword, '').strip()
                    
                    if not text_for_analysis:
                        local_fallback_response = "Please provide text or a link to analyze for fake news."
                    else:
                        message_transformed = fake_news_vectorizer.transform([text_for_analysis])
                        prediction = fake_news_classifier.predict(message_transformed)[0]
                        
                        reason_row = fake_news_df[fake_news_df['text'].str.contains(text_for_analysis, case=False, na=False)]
                        reason = reason_row.iloc[0]['reason'] if not reason_row.empty else "No specific reason from local data."
                        
                        ai_response_english = f"Local model analysis of '{text_for_analysis}': This news is likely **{prediction}**. Reason: {reason}"
                        local_fallback_response = ai_response_english
                        
                        if detected_lang != 'en':
                            final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                            return jsonify({'response': final_response})
                        else:
                            return jsonify({'response': ai_response_english})
                except Exception as e:
                    print(f"ERROR: Error during local fake news detection: {e}")
                    local_fallback_response = f"Local fake news detection failed: {e}. Falling back to LLM APIs."
            else:
                print("DEBUG: Local fake news model not used (not loaded, URL, or is a general quoted task). Falling back to LLM APIs.")
                local_fallback_response = "Local fake news model not used. Falling back to LLM APIs."
            
            try:
                prompt = f"""
                Analyze the following content for fake news. Determine if it is likely 'real' or 'fake' and provide a brief reason.
                If a URL is provided, browse its content to assess credibility.
                
                Content: "{content_to_analyze}"
                """
                ai_response_english = orchestrate_llm_calls(prompt)
                if not ai_response_english: # If LLM orchestration failed
                    ai_response_english = local_fallback_response
            except Exception as e:
                print(f"ERROR: Error calling LLM orchestration for fake news detection: {e}")
                ai_response_english = local_fallback_response

        # --- Priority 6: Code Analysis ---
        elif any(keyword in processed_message.lower() for keyword in code_keywords):
            local_fallback_response = "I couldn't perform local code analysis for your request."
            
            code_snippet, requirement_text = extract_code_and_requirement(processed_message)

            if CODE_ANALYSIS_LOADED and code_snippet and not is_general_quoted_task:
                try:
                    matched_row = None
                    for index, row in code_analysis_df.iterrows():
                        if row['code_snippet_pattern'] in code_snippet:
                            matched_row = row
                            break
                    
                    if matched_row is None:
                        for index, row in code_analysis_df.iterrows():
                            if any(kw in row['issue_or_analysis_focus'].lower() for kw in requirement_text.lower().split()):
                                matched_row = row
                                break

                    if matched_row is not None:
                        ai_response_english = (
                            f"Local Code Analysis Result:\n"
                            f"**Language:** {matched_row['code_language']}\n"
                            f"**Issue/Focus:** {matched_row['issue_or_analysis_focus']}\n"
                            f"**Guidance:** {matched_row['local_response']}"
                        )
                        local_fallback_response = ai_response_english
                        
                        if detected_lang != 'en':
                            final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                            return jsonify({'response': final_response})
                        else:
                            return jsonify({'response': ai_response_english})
                    else:
                        local_fallback_response = "No specific local code analysis found for this snippet/query. Falling back to LLM APIs."
                except Exception as e:
                    print(f"ERROR: Error during local code analysis: {e}")
                    local_fallback_response = f"Local code analysis failed: {e}. Falling back to LLM APIs."
            else:
                print("DEBUG: Local code analysis model not used (not loaded, no snippet, or is a general quoted task). Falling back to LLM APIs.")
                local_fallback_response = "Local code analysis model not used. Falling back to LLM APIs."
            
            try:
                prompt = f"""
                Analyze the following code and/or user requirement.
                If code is provided, identify its language, fix any errors, explain the fix, or analyze its purpose based on the request.
                If only a request is provided, offer general guidance.
                
                User Request: "{requirement_text}"
                Code Snippet: "{code_snippet if code_snippet else 'No code provided.'}"
                """
                ai_response_english = orchestrate_llm_calls(prompt)
                if not ai_response_english: # If LLM orchestration failed
                    ai_response_english = local_fallback_response
            except Exception as e:
                print(f"ERROR: Error calling LLM orchestration for code analysis: {e}")
                ai_response_english = local_fallback_response

        # --- Priority 7: Disease Prediction ---
        elif any(keyword in processed_message.lower() for keyword in disease_keywords):
            local_fallback_response = "I couldn't predict a disease from your symptoms locally."
            
            if DISEASE_MODEL_LOADED and not is_general_quoted_task:
                try:
                    input_symptoms_df = pd.DataFrame(0, index=[0], columns=symptom_columns_list)
                    user_symptoms_lower = processed_message.lower()
                    
                    detected_symptoms = []
                    for symptom_col in symptom_columns_list:
                        if symptom_col.replace('_', ' ') in user_symptoms_lower or \
                           symptom_col in user_symptoms_lower:
                            input_symptoms_df[symptom_col] = 1
                            detected_symptoms.append(symptom_col)
                    
                    if detected_symptoms:
                        prediction = disease_classifier.predict(input_symptoms_df)[0]
                        
                        if 'prognosis' not in disease_df.columns:
                            raise ValueError("Disease dataset missing 'prognosis' column after cleaning.")

                        medicine_suggestion = disease_df[disease_df['prognosis'] == prediction]['medicine_suggestion'].iloc[0]
                        ai_response_english = f"Based on your symptoms, you might have **{prediction}**. Suggested care: {medicine_suggestion}"
                        local_fallback_response = ai_response_english
                        
                        if detected_lang != 'en':
                            final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                            return jsonify({'response': final_response})
                        else:
                            return jsonify({'response': ai_response_english})
                except Exception as e:
                    print(f"ERROR: Error during local disease model prediction: {e}")
                    local_fallback_response = f"Local disease prediction failed: {e}. Falling back to LLM APIs."
            else:
                print("DEBUG: Local disease model not used (not loaded, or is a general quoted task). Falling back to LLM APIs.")
                local_fallback_response = "Local disease model not used. Falling back to LLM APIs."
            
            try:
                prompt = f"""
                Analyze the following symptoms and provide:
                1. A possible disease diagnosis.
                2. A brief suggestion for medicine or care.
                Symptoms: "{processed_message}"
                """
                if user_location:
                    prompt += f"\nConsider the context of {user_location} if relevant."
                ai_response_english = orchestrate_llm_calls(prompt)
                if not ai_response_english: # If LLM orchestration failed
                    ai_response_english = local_fallback_response
            except Exception as e:
                print(f"ERROR: Error calling LLM orchestration for disease prediction: {e}")
                ai_response_english = local_fallback_response

        # --- Priority 8: Study/Job Guidance ---
        elif any(keyword in processed_message.lower() for keyword in study_keywords):
            local_fallback_response = "I couldn't find local study or job guidance for your query."
            
            if STUDY_GUIDE_LOADED and not is_general_quoted_task:
                try:
                    job_role_query = ""
                    match = re.search(r'(?:for\s+a\s+|for\s+an\s+|for\s+the\s+|guide\s+for\s+|how\s+to\s+become\s+a\s+|how\s+to\s+study\s+for\s+a\s+)?([a-zA-Z\s]+?)(?:\s+job|\s+role|\s+position|\s+engineer|\s+scientist|\s+developer|\s+analyst|\s+designer)?', processed_message.lower())
                    
                    if match:
                        potential_role = match.group(1).strip()
                        if "software" in potential_role: job_role_query = "software engineer"
                        elif "data scien" in potential_role: job_role_query = "data scientist"
                        elif "machine learning" in potential_role: job_role_query = "machine learning engineer"
                        elif "web dev" in potential_role: job_role_query = "web developer"
                        elif "cybersecurity" in potential_role: job_role_query = "cybersecurity analyst"
                        elif "product man" in potential_role: job_role_query = "product manager"
                        elif "ux design" in potential_role: job_role_query = "ux designer"
                        else:
                            job_role_query = potential_role

                    if job_role_query:
                        matching_rows = study_guide_df[
                            study_guide_df['job_role'].str.contains(job_role_query, case=False, na=False)
                        ]
                        
                        if not matching_rows.empty:
                            guidance = matching_rows.iloc[0]['study_guidance']
                            application = matching_rows.iloc[0]['application_process']
                            universities = matching_rows.iloc[0]['universities']
                            
                            ai_response_english = (
                                f"For a **{matching_rows.iloc[0]['job_role']}** role:\n\n"
                                f"**Study Guidance:** {guidance}\n\n"
                                f"**Application Process:** {application}\n\n"
                                f"**Suitable Universities:** {universities}"
                            )
                            local_fallback_response = ai_response_english
                            
                            if detected_lang != 'en':
                                final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                                return jsonify({'response': final_response})
                        else:
                            return jsonify({'response': ai_response_english})
                    else:
                        local_fallback_response = f"No specific local guidance found. Falling back to LLM APIs."
                except Exception as e:
                    print(f"ERROR: Error during local study guide lookup: {e}")
                    local_fallback_response = f"Local study guide lookup failed: {e}. Falling back to LLM APIs."
            else:
                print("DEBUG: Local study guide model not used (not loaded, or is a general quoted task). Falling back to LLM APIs.")
                local_fallback_response = "Local study guide model not used. Falling back to LLM APIs."
            
            try:
                prompt = f"""
                Provide study guidance, application process, and suitable universities for the job role/study area mentioned in the following text.
                If a specific job role is not clear, provide general study advice.
                User Text: "{processed_message}"
                """
                if user_location:
                    prompt += f"\nConsider the context of {user_location} for universities and job market."
                ai_response_english = orchestrate_llm_calls(prompt)
                if not ai_response_english: # If LLM orchestration failed
                    ai_response_english = local_fallback_response
            except Exception as e:
                print(f"ERROR: Error calling LLM orchestration for study guidance: {e}")
                ai_response_english = local_fallback_response

        # --- Priority 9: Sentiment Analysis ---
        elif any(keyword in processed_message.lower() for keyword in sentiment_keywords):
            local_fallback_response = "I couldn't perform local sentiment analysis for your request."
            
            if SENTIMENT_MODEL_LOADED and not is_general_quoted_task:
                try:
                    text_for_sentiment = processed_message
                    for keyword in sentiment_keywords:
                        text_for_sentiment = text_for_sentiment.replace(keyword, '').strip()
                    
                    if not text_for_sentiment:
                        local_fallback_response = "Please provide text to analyze sentiment."
                    else:
                        message_transformed = sentiment_vectorizer.transform([text_for_sentiment])
                        prediction = sentiment_classifier.predict(message_transformed)[0]
                        
                        ai_response_english = f"Local model sentiment analysis of '{text_for_sentiment}': Your text is '{prediction}'."
                        local_fallback_response = ai_response_english
                        
                        if detected_lang != 'en':
                            final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                            return jsonify({'response': final_response})
                        else:
                            return jsonify({'response': ai_response_english})
                except Exception as e:
                    print(f"ERROR: Error during local sentiment analysis: {e}")
                    local_fallback_response = f"Local sentiment analysis failed: {e}. Falling back to LLM APIs."
            else:
                print("DEBUG: Local sentiment model not used (not loaded, or is a general quoted task). Falling back to LLM APIs.")
                local_fallback_response = "Local sentiment model not used. Falling back to LLM APIs."
            
            try:
                prompt = f"""
                Analyze the sentiment of the following text (e.g., positive, negative, neutral) and explain your reasoning.
                Text: "{processed_message}"
                """
                ai_response_english = orchestrate_llm_calls(prompt)
                if not ai_response_english: # If LLM orchestration failed
                    ai_response_english = local_fallback_response
            except Exception as e:
                print(f"ERROR: Error calling LLM orchestration for sentiment analysis: {e}")
                ai_response_english = local_fallback_response

        # --- Priority 10: Spam Classification ---
        elif any(keyword in processed_message.lower() for keyword in spam_keywords):
            local_fallback_response = "I couldn't classify your message locally."
            
            if SPAM_MODEL_LOADED and not is_general_quoted_task:
                try:
                    message_transformed = spam_vectorizer.transform([processed_message])
                    prediction = spam_classifier.predict(message_transformed)[0]
                    if prediction:
                        ai_response_english = f"Local model says: This message is likely '{prediction}'."
                        local_fallback_response = ai_response_english

                        if detected_lang != 'en':
                            final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                            return jsonify({'response': final_response})
                        else:
                            return jsonify({'response': ai_response_english})
                except Exception as e:
                    print(f"ERROR: Error during local spam model prediction: {e}")
                    local_fallback_response = f"Local spam prediction failed: {e}. Falling back to LLM APIs."
            else:
                print("DEBUG: Local spam model not used (not loaded, or is a general quoted task). Falling back to LLM APIs.")
                local_fallback_response = "Local spam model not used. Falling back to LLM APIs."
            
            try:
                prompt = f"Classify the following text as 'Spam' or 'Not Spam' and provide a brief reason:\n\n{processed_message}"
                ai_response_english = orchestrate_llm_calls(prompt)
                if not ai_response_english: # If LLM orchestration failed
                    ai_response_english = local_fallback_response
            except Exception as e:
                print(f"ERROR: Error calling LLM orchestration for spam classification: {e}")
                ai_response_english = local_fallback_response
        
        # --- Priority 11: Job Recommendations ---
        elif any(keyword in processed_message.lower() for keyword in job_recommendation_keywords):
            local_fallback_response = "I couldn't provide job recommendations locally. Falling back to LLM APIs."
            try:
                prompt = f"""
                Based on the following user-provided text, analyze the skills and qualifications.
                Suggest a list of 3 suitable job roles. For each role, provide:
                1. A brief reason why it's a good fit.
                2. 3 specific companies to consider applying to.
                3. The best time of year to apply for these types of roles.
                
                User Text: "{processed_message}"
                """
                if user_location:
                    prompt += f"\nConsider the job market and companies in {user_location}."
                ai_response_english = orchestrate_llm_calls(prompt)
                if not ai_response_english: # If LLM orchestration failed
                    ai_response_english = local_fallback_response
            except Exception as e:
                print(f"ERROR: Error calling LLM orchestration for job recommendations: {e}")
                ai_response_english = local_fallback_response
        
        # --- Priority 12: General Chat ---
        else:
            local_fallback_response = "I'm sorry, I couldn't process your request right now. Please try again later."
            try:
                prompt = f"Respond to the following user message: {processed_message}"
                ai_response_english = orchestrate_llm_calls(prompt)
                if not ai_response_english: # If LLM orchestration failed
                    ai_response_english = local_fallback_response
            except Exception as e:
                print(f"ERROR: Error calling LLM orchestration for general chat: {e}")
                ai_response_english = local_fallback_response
        
        # Step 4: Translate Response back to original language if needed (for non-translation/summarization/TTS tasks)
        # For translation/summarization/TTS tasks, the response is already formatted
        # and should not be re-translated.
        final_response = ai_response_english
        if detected_lang != 'en' and ai_response_english and \
           not (text_to_translate_raw and translation_command and target_language) and \
           not (content_to_summarize and summarization_command) and \
           not (content_for_tts_raw and tts_command):
            print(f"DEBUG: Translating English response '{ai_response_english}' back to '{detected_lang}'...")
            try:
                final_response = translate_text_llm(ai_response_english, detected_lang, 'en')
                if not final_response:
                    raise ValueError("Translation back to original language resulted in empty text.")
                print(f"DEBUG: Final translated response: '{final_response}'")
            except Exception as e:
                print(f"ERROR: Error translating response back to '{detected_lang}': {e}. Sending English response instead.")
                final_response = ai_response_english
        else:
            print(f"DEBUG: Final response (English or already formatted): '{final_response}'")
        
        return jsonify({'response': final_response})
    
    except ValueError as ve:
        print(f"ERROR: Value error occurred: {str(ve)}")
        return jsonify({'error': str(ve)}), 500
    except requests.exceptions.RequestException as e:
        print(f"ERROR: API request error occurred: {e}")
        return jsonify({'error': f"API Error: {e}"}), 500
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {str(e)}")
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        # This finally block ensures that temporary directories are cleaned up
        # if the media task was *not* a download link (which is handled by /download_media)
        # or if an error occurred before a download link could be generated.
        if media_result and media_result['type'] != 'download_link' and 'temp_dir' in media_result:
            temp_dir_to_clean = media_result['temp_dir']
            if os.path.exists(temp_dir_to_clean):
                try:
                    shutil.rmtree(temp_dir_to_clean)
                    print(f"DEBUG: Cleaned up temporary directory: {temp_dir_to_clean}")
                except OSError as e:
                    print(f"WARNING: Could not remove temporary directory {temp_dir_to_clean} in finally block: {e}")


@app.route('/test_spam_model', methods=['POST'])
def test_spam_model():
    """A dedicated endpoint to test the local spam classification model."""
    if not SPAM_MODEL_LOADED:
        return jsonify({"error": "Local spam model not loaded. Please ensure 'train_model.py' has been run."}), 500
    
    try:
        data = request.get_json()
        text_to_check = data.get('text', '')
        
        if not text_to_check:
            return jsonify({"error": "No text provided for classification."}), 400
        
        message_transformed = spam_vectorizer.transform([text_to_check])
        prediction = spam_classifier.predict(message_transformed)[0]
        
        return jsonify({
            "status": "success",
            "text": text_to_check,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during local model testing: {str(e)}"}), 500

@app.route('/test_disease_model', methods=['POST'])
def test_disease_model():
    """A dedicated endpoint to test the local disease classification model."""
    if not DISEASE_MODEL_LOADED:
        return jsonify({"error": "Local disease model not loaded. Please ensure 'train_disease_model.py' has been run."}), 500
    
    try:
        data = request.get_json()
        symptoms_text = data.get('symptoms', '')
        
        if not symptoms_text:
            return jsonify({"error": "No symptoms provided for classification."}), 400
        
        input_symptoms_df = pd.DataFrame(0, index=[0], columns=symptom_columns_list)
        symptoms_text_lower = symptoms_text.lower()

        detected_symptoms = []
        for symptom_col in symptom_columns_list:
            if symptom_col.replace('_', ' ') in symptoms_text_lower or \
               symptom_col in symptoms_text_lower:
                input_symptoms_df[symptom_col] = 1
                detected_symptoms.append(symptom_col)
        
        if not detected_symptoms:
            return jsonify({
                "status": "failure",
                "message": "No recognized symptoms found in the input for local prediction.",
                "input_text": symptoms_text
            }), 200

        prediction = disease_classifier.predict(input_symptoms_df)[0]
        if 'prognosis' not in disease_df.columns:
            raise ValueError("Disease dataset missing 'prognosis' column after cleaning.")

        medicine_suggestion = disease_df[disease_df['prognosis'] == prediction]['medicine_suggestion'].iloc[0]

        return jsonify({
            "status": "success",
            "symptoms_input": symptoms_text,
            "detected_symptoms": detected_symptoms,
            "prediction": prediction,
            "medicine_suggestion": medicine_suggestion
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during local model testing: {str(e)}"}), 500

@app.route('/test_study_guide', methods=['POST'])
def test_study_guide():
    """
    A dedicated endpoint to test the local study guidance lookup.
    """
    if not STUDY_GUIDE_LOADED:
        return jsonify({"error": "Local study guide dataset not loaded. Please ensure 'train_study_guide_model.py' has been run."}), 500
    
    try:
        data = request.get_json()
        job_role_input = data.get('job_role', '').strip().lower()
        
        if not job_role_input:
            return jsonify({"error": "No job role provided for study guidance lookup."}), 400

        matching_rows = study_guide_df[
            study_guide_df['job_role'].str.contains(job_role_input, case=False, na=False)
        ]
        
        if not matching_rows.empty:
            guidance = matching_rows.iloc[0]['study_guidance']
            application = matching_rows.iloc[0]['application_process']
            universities = matching_rows.iloc[0]['universities']
            job_role_found = matching_rows.iloc[0]['job_role']
            
            return jsonify({
                "status": "success",
                "job_role_queried": job_role_input,
                "job_role_found": job_role_found,
                "study_guidance": guidance,
                "application_process": application,
                "suitable_universities": universities
            })
        else:
            return jsonify({
                "status": "not_found",
                "message": f"No local study guidance found for '{job_role_input}'.",
                "job_role_queried": job_role_input
            }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during local study guide testing: {str(e)}"}), 500

@app.route('/test_sentiment_model', methods=['POST'])
def test_sentiment_model():
    """
    A dedicated endpoint to test the local sentiment classification model.
    """
    if not SENTIMENT_MODEL_LOADED:
        return jsonify({"error": "Local sentiment model not loaded. Please ensure 'train_sentiment_model.py' has been run."}), 500
    
    try:
        data = request.get_json()
        text_to_check = data.get('text', '')
        
        if not text_to_check:
            return jsonify({"error": "No text provided for sentiment analysis."}), 400
        
        message_transformed = sentiment_vectorizer.transform([text_to_check])
        prediction = sentiment_classifier.predict(message_transformed)[0]
        
        return jsonify({
            "status": "success",
            "text": text_to_check,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during local sentiment model testing: {str(e)}"}), 500

@app.route('/test_code_analysis', methods=['POST'])
def test_code_analysis():
    """
    A dedicated endpoint to test the local code analysis lookup.
    """
    if not CODE_ANALYSIS_LOADED:
        return jsonify({"error": "Local code analysis dataset not loaded. Please ensure 'train_code_analysis_data.py' has been run."}), 500
    
    try:
        data = request.get_json()
        user_input_text = data.get('text', '').strip()
        
        if not user_input_text:
            return jsonify({"error": "No text provided for code analysis."}), 400

        code_snippet, requirement_text = extract_code_and_requirement(user_input_text)
        
        matched_row = None
        response_message = ""

        if code_snippet:
            for index, row in code_analysis_df.iterrows():
                if row['code_snippet_pattern'] in code_snippet:
                    matched_row = row
                    break
        
        if matched_row is None and requirement_text:
            for index, row in code_analysis_df.iterrows():
                if any(kw in row['issue_or_analysis_focus'].lower() for kw in requirement_text.lower().split()):
                    matched_row = row
                    break

        if matched_row is not None:
            response_message = (
                f"Local Code Analysis Result:\n"
                f"**Language:** {matched_row['code_language']}\n"
                f"**Issue/Focus:** {matched_row['issue_or_analysis_focus']}\n"
                f"**Guidance:** {matched_row['local_response']}"
            )
            return jsonify({
                "status": "success",
                "input_text": user_input_text,
                "code_snippet": code_snippet,
                "requirement_text": requirement_text,
                "local_response": response_message
            })
        else:
            return jsonify({
                "status": "not_found",
                "message": "No local code analysis found for this input.",
                "input_text": user_input_text,
                "code_snippet": code_snippet,
                "requirement_text": requirement_text
            }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during local code analysis testing: {str(e)}"}), 500

@app.route('/test_fake_news_model', methods=['POST'])
def test_fake_news_model():
    """
    A dedicated endpoint to test the local fake news detection model.
    """
    if not FAKE_NEWS_LOADED:
        return jsonify({"error": "Local fake news model not loaded. Please ensure 'train_fake_news_model.py' has been run."}), 500
    
    try:
        data = request.get_json()
        user_input_text = data.get('text', '').strip()
        
        if not user_input_text:
            return jsonify({"error": "No text provided for fake news analysis."}), 400

        content_to_analyze, original_query_text, content_type = extract_url_or_content_for_fake_news(user_input_text)

        if content_type == "url":
            return jsonify({
                "status": "url_detected",
                "message": "Local model cannot analyze URLs. This would trigger API fallback in chat.",
                "url": content_to_analyze,
                "original_query": original_query_text
            }), 200
        else:
            text_for_analysis = content_to_analyze
            fake_news_keywords = ["fake", "news", "is fake", "is news", "fact check", "real or fake", "verify news"]
            for keyword in fake_news_keywords:
                text_for_analysis = text_for_analysis.replace(keyword, '').strip()
            
            if not text_for_analysis:
                return jsonify({
                    "status": "failure",
                    "message": "No meaningful text provided for local fake news analysis after keyword removal.",
                    "input_text": user_input_text
                }), 200

            message_transformed = fake_news_vectorizer.transform([text_for_analysis])
            prediction = fake_news_classifier.predict(message_transformed)[0]
            
            reason = "No specific reason from local data."
            matching_rows = fake_news_df[fake_news_df['text'].str.contains(text_for_analysis, case=False, na=False)]
            if not matching_rows.empty:
                reason = matching_rows.iloc[0]['reason']

            return jsonify({
                "status": "success",
                "input_text": user_input_text,
                "text_analyzed": text_for_analysis,
                "prediction": prediction,
                "reason": reason
            })

    except Exception as e:
        return jsonify({"error": f"An error occurred during local fake news model testing: {str(e)}"}), 500

@app.route('/test_interview_qa', methods=['POST'])
def test_interview_qa():
    """
    A dedicated endpoint to test the local interview Q&A lookup.
    """
    if not INTERVIEW_QA_LOADED:
        return jsonify({"error": "Local interview Q&A dataset not loaded."}), 500
    
    try:
        data = request.get_json()
        job_role_input = data.get('job_role', '').strip().lower()
        difficulty_input = data.get('difficulty', '').strip().lower()
        num_questions_input = int(data.get('num_questions', 1))

        if not job_role_input or not difficulty_input:
            return jsonify({"error": "Job role and difficulty are required for testing."}), 400

        # Filter by job role (using contains for flexibility)
        filtered_by_role = interview_qa_df[
            interview_qa_df['job_role'].str.contains(job_role_input, case=False, na=False)
        ]
        
        # Filter by difficulty
        filtered_by_difficulty = filtered_by_role[
            filtered_by_role['difficulty'] == difficulty_input
        ]

        questions = []
        if not filtered_by_difficulty.empty:
            available_questions = filtered_by_difficulty['question'].tolist()
            import random
            if len(available_questions) > num_questions_input:
                questions.extend(random.sample(available_questions, num_questions_input))
            else:
                questions.extend(available_questions)
            
            return jsonify({
                "status": "success",
                "job_role_queried": job_role_input,
                "difficulty_queried": difficulty_input,
                "num_questions_requested": num_questions_input,
                "questions_found": questions,
                "source": "local_dataset"
            })
        else:
            return jsonify({
                "status": "not_found",
                "message": f"No local interview questions found for '{job_role_input}' at '{difficulty_input}' difficulty.",
                "job_role_queried": job_role_input,
                "difficulty_queried": difficulty_input,
                "num_questions_requested": num_questions_input,
                "source": "local_dataset"
            }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during local interview Q&A testing: {str(e)}"}), 500














@app.route("/page-content/<page_name>", methods=["GET"])
def get_page_content(page_name):
    pages = {
        "home": "Welcome to the Home page!",
        "about": "This is the About Us page.",
        "docs": "Here is the documentation content.",
        "contact": "You can contact us at support@example.com",
        "faq": "Here are some frequently asked questions..."
    }
    if page_name.lower() in pages:
        return jsonify({"success": True, "content": pages[page_name.lower()]})
    else:
        return jsonify({"success": False, "error": "Page not found"}), 404




@app.errorhandler(Exception)
def handle_error(e):
    # Log to SQLite
    conn = sqlite3.connect("db/bot_errors.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("INSERT INTO errors (error_message) VALUES (?)", (str(e),))
    conn.commit()
    conn.close()

    # Return generic message to user
    return jsonify({"error": "Something went wrong, please try again later."}), 500


















if __name__ == '__main__':
    # Initial check for API keys
    print("\n--- API Key Status ---")
    for llm_name, key in API_KEYS.items():
        if not key: # Check if key is empty string, which is the default if not found in .env
            print(f"WARNING: API key for '{llm_name}' is not set in .env. API calls to this service will fail.")
        else:
            print(f"API key for '{llm_name}' appears to be set.")
    print("----------------------\n")
    
    app.run(debug=True, port=5000)
