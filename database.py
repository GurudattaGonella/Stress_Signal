import sqlite3
import hashlib
from datetime import datetime

# Name of the database file
DB_NAME = "stress_signal.db"

def init_db():
    """
    Creates the necessary tables if they don't exist.
    Run this once at the start of your app.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. Create USERS Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    # 2. Create REPORTS Table
    # Stores the summary of a session (BPM, Stress, Analysis)
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            avg_bpm INTEGER,
            avg_stress INTEGER,
            stress_trend TEXT,
            ai_analysis TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # 3. Create CHAT MEMORY Table (For your future Chatbot)
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT, -- 'user' or 'bot'
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")

# --- USER FUNCTIONS ---

def register_user(username, password):
    """Adds a new user with a hashed password."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Simple hash for security (SHA256)
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pwd_hash))
        conn.commit()
        print(f"✅ User '{username}' registered.")
        return True
    except sqlite3.IntegrityError:
        print(f"❌ User '{username}' already exists.")
        return False
    finally:
        conn.close()

def login_user(username, password):
    """Returns User ID if credentials match, else None."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT id FROM users WHERE username=? AND password_hash=?", (username, pwd_hash))
    user = c.fetchone()
    conn.close()
    
    if user:
        return user[0] # Return the User ID
    return None

# --- REPORT FUNCTIONS ---

def save_report(user_id, avg_bpm, avg_stress, stress_trend="Stable", ai_analysis="No analysis"):
    """Saves a session report to the DB."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO reports (user_id, avg_bpm, avg_stress, stress_trend, ai_analysis)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, avg_bpm, avg_stress, stress_trend, ai_analysis))
    
    conn.commit()
    conn.close()
    print("✅ Report saved to database.")

def get_last_reports(user_id, limit=5):
    """Fetches the last N reports for the Analyzer."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''
        SELECT timestamp, avg_bpm, avg_stress, stress_trend, ai_analysis 
        FROM reports 
        WHERE user_id=? 
        ORDER BY id DESC 
        LIMIT ?
    ''', (user_id, limit))
    
    data = c.fetchall()
    conn.close()
    return data # Returns a list of rows

# database.py

def get_recent_reports(limit=10):
    """
    Fetches the last 'limit' reports from the database.
    Returns a list of dictionaries formatted for the frontend.
    """
    # specific import inside function to avoid circular dependency issues
    import sqlite3 
    
    conn = sqlite3.connect('stress_signal.db') # Make sure this matches your DB name
    cursor = conn.cursor()
    
    # adjust column names (id, timestamp, bpm, stress) to match your actual table
    cursor.execute("""
        SELECT id, timestamp, avg_bpm, avg_stress, ai_analysis 
        FROM reports 
        ORDER BY id DESC 
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert database rows to JSON-friendly list
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "date": row[1],
            "bpm": row[2],
            "stress": row[3],
            "message": row[4],
            # Assuming PDF name is generated from ID or Date, or you can store path in DB
            "download_url": f"/download_report?file=report_{row[0]}.pdf" 
        })
        
    return results