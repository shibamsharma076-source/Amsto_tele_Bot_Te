import sqlite3
from pymongo import MongoClient
from datetime import datetime
import bcrypt

# ====== CONFIG ======
USE_SQLITE = True  # Set to False if using MongoDB Atlas
SQLITE_DB_PATH = "users.db"

USE_MONGODB = True  # Set to True if using MongoDB Atlas
MONGO_URI = "mongodb+srv://shibams076:fX07xQEfV1VwZZAb@sjcollect.gn0apns.mongodb.net/ai_bot?retryWrites=true&w=majority"
MONGO_DB_NAME = "ai_bot"
MONGO_COLLECTION = "Users"
# ====================

def hash_password(password: str) -> str:
    """Hash password with bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")

def create_admin_sqlite(email: str, password: str):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        github_oauth INTEGER DEFAULT 0,
        role TEXT DEFAULT 'user',
        created_at TEXT
    )
    """)

    password_hash = hash_password(password)

    try:
        cur.execute("""
        INSERT INTO users (email, password_hash, github_oauth, role, created_at)
        VALUES (?, ?, ?, ?, ?)
        """, (email, password_hash, 0, "admin", datetime.utcnow().isoformat()))
        conn.commit()
        print(f"✅ Admin '{email}' created successfully in SQLite.")
    except sqlite3.IntegrityError:
        print(f"⚠ User '{email}' already exists in SQLite.")
    finally:
        conn.close()

def create_admin_mongo(email: str, password: str):
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION]

    password_hash = hash_password(password)

    if collection.find_one({"email": email}):
        print(f"⚠ User '{email}' already exists in MongoDB.")
        return

    collection.insert_one({
        "email": email,
        "password_hash": password_hash,
        "github_oauth": False,
        "role": "admin",
        "created_at": datetime.utcnow().isoformat()
    })

    print(f"✅ Admin '{email}' created successfully in MongoDB Atlas.")

if __name__ == "__main__":
    admin_email = input("Enter admin email: ").strip()
    admin_password = input("Enter admin password: ").strip()

    if USE_SQLITE:
        create_admin_sqlite(admin_email, admin_password)

    if USE_MONGODB:
        create_admin_mongo(admin_email, admin_password)
