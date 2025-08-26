#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Bot bridge for your Flask backend.

✅ Mirrors your existing frontend → backend contract:
   - POST {FLASK_BASE_URL}/chat with form-data {"message": <text>}
   - Optional header: "X-Auth-Token": <token>
   - Handles response fields:
        - response (str): main text
        - download_info { type, download_url, filename }
          - type == "audio_tts" | "media_download"
        - qr_image_base64, barcode_image_base64, uploaded_image_base64
        - recipe_image_url, recipe_video_url

Commands:
  /start – help & setup
  /login <email> <password> – authenticate via Flask /login
  /register <full_name>|<dob YYYY-MM-DD>|<username>|<email>|<password>
  /logout – delete auth token
  (Send any text or URL+\"command\" to trigger features same as Streamlit)

Environment variables:
  TELEGRAM_BOT_TOKEN   – your Telegram bot token
  FLASK_BASE_URL       – e.g. http://localhost:5000
  REQUEST_TIMEOUT_SEC  – optional (default 30)

Run:
  pip install python-telegram-bot==21.6 requests python-dotenv
  python telegram_bot.py
"""
from email.mime import application
import os
import re
import json
import base64
import logging
from io import BytesIO
from typing import Dict, Optional
import sqlite3
import requests
from sqlalchemy import update
from telegram import Update, InputFile
from telegram import (
    Update, InputFile, ReplyKeyboardMarkup, KeyboardButton
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters, ConversationHandler
)
from pymongo import MongoClient
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


# ---------- Config ----------
# FLASK_BASE_URL = os.getenv("FLASK_BASE_URL", "http://localhost:5000")
FLASK_BASE_URL = os.getenv("FLASK_BASE_URL", "http://13.60.240.108:5000")
CHAT_ENDPOINT = f"{FLASK_BASE_URL}/chat"
LOGIN_ENDPOINT = f"{FLASK_BASE_URL}/login"
REGISTER_ENDPOINT = f"{FLASK_BASE_URL}/register"
LOGOUT_ENDPOINT = f"{FLASK_BASE_URL}/logout"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SEC", "60"))

# In-memory auth tokens keyed by Telegram user id
USER_TOKENS: Dict[int, str] = {}
USER_EMAILS: Dict[int, str] = {}
USER_ROLES: Dict[int, str] = {}





# DB Setup (SQLite + Mongo)
# -----------------------------
BOT_SQLITE_DB = "db/bot_users.db"
BOT_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(BOT_MONGO_URI)
mongo_db = mongo_client["telegram_bot"]   # database name
mongo_users = mongo_db["users"]           # collection name







# --- Security Scanner ---
SQLI_PATTERNS = [
    r"(\bor\b|\band\b).*(=|like)",  # OR/AND + condition
    r"(union(\s+all)?\s+select)",   # UNION SELECT
    r"(--|#)",                      # SQL comment
    r"(drop|insert|update|delete|truncate)\s",  # DDL/DML
]

def looks_like_sql_injection(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    for pattern in SQLI_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False

def looks_like_virus_file(file_name: str) -> bool:
    if not file_name:
        return False
    # Add more dangerous extensions if you like
    blocked_exts = [".exe", ".bat", ".cmd", ".vbs", ".js", ".sh", ".scr", ".pif", ".msi"]
    return any(file_name.lower().endswith(ext) for ext in blocked_exts)













def init_bot_sqlite():
    conn = sqlite3.connect(BOT_SQLITE_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER UNIQUE,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            phone TEXT,
            email TEXT,
            latitude REAL,
            longitude REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_bot_sqlite()

def save_user(telegram_id, username, first_name, last_name, phone=None, email=None, lat=None, lon=None):
    # Save SQLite
    conn = sqlite3.connect(BOT_SQLITE_DB)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO users (telegram_id, username, first_name, last_name, phone, email, latitude, longitude, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (telegram_id, username, first_name, last_name, phone, email, lat, lon, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    # Save Mongo
    mongo_users.update_one(
        {"telegram_id": telegram_id},
        {"$set": {
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
            "phone": phone,
            "email": email,
            "latitude": lat,
            "longitude": lon,
            "created_at": datetime.utcnow()
        }},
        upsert=True
    )











BOT_DB = "db/bot_errors.db"

def log_bot_error(user_id, user_input, error_message):
    conn = sqlite3.connect(BOT_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_input TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("INSERT INTO errors (user_id, user_input, error_message) VALUES (?, ?, ?)",
                (user_id, user_input, error_message))
    conn.commit()
    conn.close()














logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------- Helpers ----------
# def _auth_headers(user_id: int) -> Dict[str, str]:
#     token = USER_TOKENS.get(user_id)
#     return {"X-Auth-Token": token} if token else {}

def _split_text(text: str, limit: int = 4096):
    # Telegram message limit
    return [text[i:i+limit] for i in range(0, len(text), limit)]

async def _send_long_text(update: Update, text: str):
    for chunk in _split_text(text):
        await update.effective_chat.send_message(chunk, disable_web_page_preview=True)

def _full_url(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return f"{FLASK_BASE_URL}{path_or_url}"

def _download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content








def user_exists(telegram_id):
    # Check SQLite
    conn = sqlite3.connect(BOT_SQLITE_DB)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE telegram_id = ?", (telegram_id,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists












ASK_EMAIL = 1

# ---------- Commands ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    msg = [
        "👋 Hi! I'm Amsto Telegram bot.",
        "I Am an AI assistant cum chatbot & Telegram Ai Tool bot.",
        "",
        "I Do This Tasks ",
        "Qualification-Based Job Recommendation",
        "Job-to-Qualification Recommendation",
        "Disease Prediction",
        "Spam and UPI Fraud Detection",
        "Multilingual Support",
        "Fake News Detection",
        "Sentiment Analysis of Social Media Data",
        "Social Media Downloader",
        "Interview Question Generator",
        "Text Translation",
        "Text Summarization with Limit Control",
        "Text-to-Speech (TTS) with Role-Based Voices",
        "Plagiarism Checker",
        "Code Generation",
        "QR Code Generator/Reader",
        "Recipe Generator",
        "Text Rewriter/Paraphraser",
        "API Tester",
        "Grammar and Spell Checker",
        "",
        "• Send any message ",
        "• Use patterns like:",
        '    https://youtube.com/.... "download video"',
        '    "Translate to Hindi" <your text>',
        '    Symptoms "Disease"',
        '    Job Role "Qualification"',
        "",
        
    ]
    await _send_long_text(update, "\n".join(msg))

    user = update.effective_user

    # ✅ Check if user already exists
    if user_exists(user.id):
        await update.message.reply_text(
            f"👋 Welcome back {user.first_name}! 🎉\n"
        )
        return ConversationHandler.END

    # Save base info for new users
    save_user(user.id, user.username, user.first_name, user.last_name)

    # Ask for phone + location
    keyboard = ReplyKeyboardMarkup(
        [[KeyboardButton("📱 Share Phone Number", request_contact=True)],
         [KeyboardButton("📍 Share Location", request_location=True)]],
        resize_keyboard=True,
        one_time_keyboard=True
    )

    await update.message.reply_text(
        f"👋 Hi {user.first_name}! I saved your Telegram ID.\n\n"
        f"Please share your phone number and location 👇",
        reply_markup=keyboard
    )

    # Ask email
    await update.message.reply_text("📧 Please type your email address:")
    # Insert user basic info only once
    conn = sqlite3.connect("db/bot_users.db")
    cur = conn.cursor()
    cur.execute("""INSERT OR IGNORE INTO users
                   (telegram_id, username, first_name, last_name, created_at)
                   VALUES (?, ?, ?, ?, datetime('now'))""",
                (update.effective_user.id, update.effective_user.username,
                 update.effective_user.first_name, update.effective_user.last_name))
    conn.commit()
    conn.close()
    return ASK_EMAIL
    






def update_user_phone(telegram_id, phone):
    conn = sqlite3.connect("db/bot_users.db")
    cur = conn.cursor()
    cur.execute("UPDATE users SET phone = ? WHERE telegram_id = ?", (phone, telegram_id))
    conn.commit()
    conn.close()

def update_user_location(telegram_id, lat, lon):
    conn = sqlite3.connect("db/bot_users.db")
    cur = conn.cursor()
    cur.execute("UPDATE users SET latitude = ?, longitude = ? WHERE telegram_id = ?", (lat, lon, telegram_id))
    conn.commit()
    conn.close()

def update_user_email(telegram_id, email):
    conn = sqlite3.connect("db/bot_users.db")
    cur = conn.cursor()
    cur.execute("UPDATE users SET email = ? WHERE telegram_id = ?", (email, telegram_id))
    conn.commit()
    conn.close()

async def contact_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    contact = update.message.contact
    if contact:
        update_user_phone(update.effective_user.id, contact.phone_number)
        await update.message.reply_text("✅ Phone number saved!")

async def location_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    loc = update.message.location
    if loc:
        update_user_location(update.effective_user.id, loc.latitude, loc.longitude)
        await update.message.reply_text("✅ Location saved!")


async def email_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = update.message.text.strip()

    if "@" in text and "." in text:
        update_user_email(user.id, text)
        await update.message.reply_text("✅ Email saved!")
        return ConversationHandler.END   # ✅ End conversation, go back to normal bot flow
    else:
        await update.message.reply_text("⚠️ That doesn’t look like a valid email. Try again.")
        return ASK_EMAIL  # Still waiting for valid email





























async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if doc:
        file_name = doc.file_name or ""
        if looks_like_virus_file(file_name):
            log_bot_error(update.effective_user.id, file_name, "Blocked: Dangerous file type")
            return await update.message.reply_text("❌ File blocked: Potentially dangerous file type.")

        # If safe, let’s just save or forward the file to backend
        file = await doc.get_file()
        file_bytes = await file.download_as_bytearray()
        await update.message.reply_text(f"✅ File {file_name} received safely ({len(file_bytes)} bytes).")



# "Authentication:",
#         "  /login <email> <password>",
#         "  /logout",
#         "  /register <full_name>|<dob YYYY-MM-DD>|<username>|<email>|<password>",

# async def login(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     # /login email password
#     if len(context.args) < 2:
#         return await update.message.reply_text("Usage: /login <email> <password>")
#     email = context.args[0]
#     password = " ".join(context.args[1:])
#     try:
#         resp = requests.post(
#             LOGIN_ENDPOINT,
#             json={"email": email, "password": password},
#             timeout=REQUEST_TIMEOUT,
#         )
#         if resp.status_code == 200:
#             data = resp.json()
#             USER_TOKENS[update.effective_user.id] = data.get("token", "")
#             USER_EMAILS[update.effective_user.id] = data.get("email", email)
#             USER_ROLES[update.effective_user.id] = data.get("role", "user")
#             await update.message.reply_text(f"✅ Logged in as {USER_EMAILS[update.effective_user.id]} (role: {USER_ROLES[update.effective_user.id]})")
#         else:
#             try:
#                 err = resp.json().get("error", resp.text)
#             except Exception:
#                 err = resp.text
#             await update.message.reply_text(f"❌ Login failed: {err}")
#     except Exception as e:
#         await update.message.reply_text(f"⚠️ Login error: {e}")

# async def register(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     # /register Full Name|YYYY-MM-DD|username|email|password
#     raw = " ".join(context.args)
#     parts = raw.split("|")
#     if len(parts) != 5:
#         return await update.message.reply_text("Usage:\n/register <full_name>|<dob YYYY-MM-DD>|<username>|<email>|<password>")
#     payload = {
#         "full_name": parts[0].strip(),
#         "dob": parts[1].strip(),
#         "username": parts[2].strip(),
#         "email": parts[3].strip(),
#         "password": parts[4].strip(),
#     }
#     try:
#         resp = requests.post(REGISTER_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
#         if resp.status_code in (200, 201):
#             await update.message.reply_text("✅ Registration successful. Now run /login <email> <password>.")
#         else:
#             try:
#                 err = resp.json().get("error", resp.text)
#             except Exception:
#                 err = resp.text
#             await update.message.reply_text(f"❌ Registration failed: {err}")
#     except Exception as e:
#         await update.message.reply_text(f"⚠️ Registration error: {e}")

# async def logout(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     try:
#         # Best-effort logout call; backend may invalidate token
#         requests.post(LOGOUT_ENDPOINT, headers=_auth_headers(update.effective_user.id), timeout=10)
#     except Exception:
#         pass
#     USER_TOKENS.pop(update.effective_user.id, None)
#     USER_EMAILS.pop(update.effective_user.id, None)
#     USER_ROLES.pop(update.effective_user.id, None)
#     await update.message.reply_text("👋 Logged out.")

# ---------- Message handler ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text or ""

    # 🚨 Security check
    if looks_like_sql_injection(user_text):
        log_bot_error(update.effective_user.id, user_text, "Blocked: SQL Injection attempt")
        return await update.message.reply_text("❌ Your message was blocked (possible SQL Injection).")


    try:
        resp = requests.post(
            CHAT_ENDPOINT,
            data={"message": user_text},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ReadTimeout:
    # Timeout → friendly message
        return await update.message.reply_text("⏳ Request timed out, please try again.")
    except requests.exceptions.ConnectionError:
        # Server not reachable
        return await update.message.reply_text(f"⚠️ Cannot reach backend at {FLASK_BASE_URL}. Please check if it’s running.")
    except Exception as e:
        # Log error internally but don’t show raw Python error to user
        log_bot_error(update.effective_user.id, user_text, str(e))
        return await update.message.reply_text("⚠️ An unexpected error occurred, please try again.")



            # headers=_auth_headers(update.effective_user.id),

    # if resp.status_code != 200:
    #     try:
    #         err = resp.json().get("error", resp.text)
    #     except Exception:
    #         err = resp.text
    #     return await update.message.reply_text(f"❌ Backend error: {err}")

    if resp.status_code != 200:
        try:
            err = resp.json().get("error", resp.text)
        except Exception:
            err = resp.text
            # log error in local DB
            log_bot_error(update.effective_user.id, user_text, f"Backend error {resp.status_code}: {err}")
        return await update.message.reply_text("⚠️ Something went wrong, please try again.")

    # data = resp.json()

    try:
        data = resp.json()
    except Exception as e:
        log_bot_error(update.effective_user.id, user_text, f"JSON parse error: {e}")
        return await update.message.reply_text("⚠️ Something went wrong, please try again.")

    if "error" in data:
        # backend sent a download error
        log_bot_error(update.effective_user.id, user_text, data["error"])
        return await update.message.reply_text("⚠️ Download failed, please try again with a different link.")



    download_info = data.get("download_info")
    if isinstance(download_info, dict) and "download_url" in download_info:
        try:
            file_bytes = _download_bytes(_full_url(download_info["download_url"]))
            bio = BytesIO(file_bytes)
            bio.name = download_info.get("filename", "file.bin")

            if download_info.get("type") == "audio_tts":
                await update.effective_chat.send_audio(audio=InputFile(bio), filename=bio.name, caption="Generated TTS")
            else:
                await update.effective_chat.send_document(document=InputFile(bio), filename=bio.name)
        except Exception as e:
            log_bot_error(update.effective_user.id, user_text, f"Download error: {e}")
            await update.message.reply_text("⚠️ Something went wrong, please try again.")







    # 1) Send main text
    main_text = data.get("response") or "No response from API."
    await _send_long_text(update, main_text)

    # 2) Recipe image & video links (optional)
    recipe_image_url = data.get("recipe_image_url")
    if recipe_image_url:
        try:
            if recipe_image_url.startswith("/"):
                recipe_image_url = _full_url(recipe_image_url)
            await update.effective_chat.send_photo(recipe_image_url)
        except Exception as e:
            logger.warning(f"Failed to send recipe image: {e}")

    recipe_video_url = data.get("recipe_video_url")
    if recipe_video_url:
        await update.effective_chat.send_message(f"▶️ {recipe_video_url}")

    # 3) QR/Barcode/Uploaded images (base64)
    for key in ("qr_image_base64", "barcode_image_base64", "uploaded_image_base64"):
        b64 = data.get(key)
        if b64:
            try:
                img_bytes = base64.b64decode(b64)
                bio = BytesIO(img_bytes)
                bio.name = f"{key}.png"
                await update.effective_chat.send_photo(photo=InputFile(bio))
            except Exception as e:
                logger.warning(f"Failed to decode/send {key}: {e}")

    # 4) download_info (TTS audio or general media)
    download_info = data.get("download_info")
    if isinstance(download_info, dict) and "download_url" in download_info:
        dl_type = download_info.get("type")
        filename = download_info.get("filename", "file.bin")
        url = _full_url(download_info["download_url"])
        try:
            file_bytes = _download_bytes(url)
        except Exception as e:
            return await update.effective_chat.send_message(
                f"⚠️ I prepared a file but couldn't fetch it automatically.\nManual link: {url}"
            )
        bio = BytesIO(file_bytes)
        bio.name = filename

        if dl_type == "audio_tts":
            # Telegram accepts voice (.ogg) or audio; we'll send as audio file
            await update.effective_chat.send_audio(audio=InputFile(bio), filename=filename, caption="Generated TTS")
        else:
            await update.effective_chat.send_document(document=InputFile(bio), filename=filename)

# ---------- App bootstrap ----------
def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN env var")

    application = Application.builder().token(token).build()


    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_EMAIL: [MessageHandler(filters.TEXT & ~filters.COMMAND, email_handler)]
        },
        fallbacks=[]
    )


    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.CONTACT, contact_handler))
    application.add_handler(MessageHandler(filters.LOCATION, location_handler))

    application.add_handler(CommandHandler("start", start))
    # application.add_handler(CommandHandler("login", login))
    # application.add_handler(CommandHandler("register", register))
    # application.add_handler(CommandHandler("logout", logout))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, file_handler))


    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
