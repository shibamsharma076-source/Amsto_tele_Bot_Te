import sqlite3

# Connect (will create the file if it doesn’t exist)
conn = sqlite3.connect("bot_errors.db")
cur = conn.cursor()

# Create the errors table
cur.execute("""
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    user_input TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("✅ errors table created in bot_errors.db")
