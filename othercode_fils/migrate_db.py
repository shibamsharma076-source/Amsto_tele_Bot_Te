import sqlite3
from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()

# Replace with your MongoDB connection string
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.get_database("ai_bot") # Choose a name for your new database

def migrate_users():
    conn = sqlite3.connect("db/bot_users.db")
    cursor = conn.cursor()
    users = cursor.execute("SELECT * FROM users").fetchall()
    user_collection = db.users
    for user in users:
        # Assuming the columns are in a fixed order based on your code
        telegram_id, username, first_name, last_name, phone, email, latitude, longitude, created_at, another_variable = user
        user_doc = {
            "telegram_id": telegram_id,
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
            "phone": phone,
            "email": email,
            "latitude": latitude,
            "longitude": longitude,
            "created_at": created_at
        }
        user_collection.insert_one(user_doc)
    conn.close()
    print("User data migrated successfully.")

def migrate_errors():
    conn = sqlite3.connect("db/bot_errors.db")
    cursor = conn.cursor()
    errors = cursor.execute("SELECT * FROM errors").fetchall()
    error_collection = db.errors
    for error in errors:
        id, error_message, created_at, extra_data, *_ = error
        error_doc = {
            "error_message": error_message,
            "created_at": created_at
        }
        error_collection.insert_one(error_doc)
    conn.close()
    print("Error data migrated successfully.")

if __name__ == "__main__":
    migrate_users()
    migrate_errors()