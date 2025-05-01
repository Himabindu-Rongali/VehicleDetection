from datetime import datetime
import sqlite3

def init_db():
    conn = sqlite3.connect('vehicles.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT,
        timestamp TEXT,
        status TEXT
    )''')
    conn.commit()
    conn.close()

def log_vehicle(plate, status):
    conn = sqlite3.connect('vehicles.db')
    c = conn.cursor()
    c.execute("INSERT INTO logs (plate, timestamp, status) VALUES (?, ?, ?)",
              (plate, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status))
    conn.commit()
    conn.close()
