import sqlite3

conn = sqlite3.connect("bot_users.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    input_text TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
print("✅ errors table created")
