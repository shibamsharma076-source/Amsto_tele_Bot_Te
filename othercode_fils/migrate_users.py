import sqlite3

conn = sqlite3.connect("bot_users.db")
cur = conn.cursor()

for column in ["phone TEXT", "email TEXT", "latitude REAL", "longitude REAL"]:
    try:
        cur.execute(f"ALTER TABLE users ADD COLUMN {column}")
        print(f"Added column {column}")
    except sqlite3.OperationalError:
        print(f"Column {column.split()[0]} already exists, skipping.")

conn.commit()
conn.close()
print("✅ Migration complete.")

