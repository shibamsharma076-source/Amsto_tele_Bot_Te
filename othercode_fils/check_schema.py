import sqlite3

conn = sqlite3.connect("bot_users.db")
cur = conn.cursor()

cur.execute("PRAGMA table_info(users)")
print(cur.fetchall())

conn.close()
