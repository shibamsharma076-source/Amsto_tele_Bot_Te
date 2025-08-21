from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import os
from dotenv import load_dotenv
load_dotenv()
FLASK_API_URL = "http://localhost:5000/chat"  # your Flask backend
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am your AI Assistant Bot 🤖. Send me any query!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    try:
        resp = requests.post(FLASK_API_URL, json={"message": user_input})
        if resp.status_code == 200:
            data = resp.json()
            if "file_url" in data:  # If Flask returned a file (QR, audio, etc.)
                file_path = data["file_url"]
                await update.message.reply_document(InputFile(file_path))
            else:
                reply_text = data.get("response", "Sorry, no response.")
                await update.message.reply_text(reply_text)
        else:
            await update.message.reply_text("Backend error. Try again later.")
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
