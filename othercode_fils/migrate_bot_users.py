import sqlite3

conn = sqlite3.connect("bot_users.db")
cur = conn.cursor()

# Add missing columns if not already there
try:
    cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
except Exception as e:
    print("Skipping email column:", e)

try:
    cur.execute("ALTER TABLE users ADD COLUMN latitude REAL")
except Exception as e:
    print("Skipping latitude column:", e)

try:
    cur.execute("ALTER TABLE users ADD COLUMN longitude REAL")
except Exception as e:
    print("Skipping longitude column:", e)

conn.commit()
conn.close()

print("✅ Migration complete. Now your bot_users.db has email, latitude, longitude columns.")
