import sqlite3
import os

old_db = "0users.db"       # your current DB
new_db = "users.db"   # new DB with correct schema

# 1. Connect to old DB
old_conn = sqlite3.connect(old_db)
old_cur = old_conn.cursor()

# Fetch id, email, password_hash, role
old_cur.execute("SELECT id, email, password_hash, role FROM users")
rows = old_cur.fetchall()
old_conn.close()

# 2. Create new DB with correct schema
if os.path.exists(new_db):
    os.remove(new_db)

new_conn = sqlite3.connect(new_db)
new_cur = new_conn.cursor()

new_cur.execute("""
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT,
    dob TEXT,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    password_hash TEXT,
    role TEXT DEFAULT 'user'
)
""")

# 3. Insert migrated data with unique usernames
for row in rows:
    old_id, email, password_hash, role = row

    # Generate username from email
    if email:
        base_username = email.split("@")[0]
    else:
        base_username = f"user{old_id}"

    username = base_username

    try:
        new_cur.execute("""
            INSERT INTO users (id, full_name, dob, username, email, password_hash, role)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (old_id, "", "", username, email, password_hash, role or "user"))
    except sqlite3.IntegrityError:
        # If username already exists, append user ID
        username = f"{base_username}_{old_id}"
        new_cur.execute("""
            INSERT INTO users (id, full_name, dob, username, email, password_hash, role)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (old_id, "", "", username, email, password_hash, role or "user"))

new_conn.commit()
new_conn.close()

print(f"✅ Migration complete. Created {new_db} with {len(rows)} users migrated.")
