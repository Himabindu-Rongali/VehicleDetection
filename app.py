from flask import Flask, render_template, request
import cv2
import torch
import easyocr
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load OCR
reader = easyocr.Reader(['en'])

# Setup SQLite database
conn = sqlite3.connect('vehicles.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT,
                timestamp TEXT,
                status TEXT
            )''')
conn.commit()

# Load authorized plates
def load_authorized_plates():
    try:
        with open('authorized_vehicles.txt', 'r') as f:
            return [line.strip().upper() for line in f]
    except FileNotFoundError:
        return []

authorized_plates = load_authorized_plates()

# Log to text file
def log_to_file(plate, timestamp, status):
    with open("detection_log.txt", "a") as f:
        f.write(f"{timestamp} | Plate: {plate} | Status: {status}\n")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded image
        file = request.files['image']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load and process image
        img = cv2.imread(filepath)
        results = model(filepath)
        predictions = results.xyxy[0]

        found_plates = []

        for *box, conf, cls in predictions:
            x1, y1, x2, y2 = map(int, box)
            cropped = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            text = reader.readtext(gray)

            for result in text:
                plate = result[1].strip().upper()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status = 'Authorized' if plate in authorized_plates else 'Unauthorized'

                # Log to database
                c.execute("INSERT INTO logs (plate, timestamp, status) VALUES (?, ?, ?)",
                          (plate, timestamp, status))
                conn.commit()

                # Log to file
                log_to_file(plate, timestamp, status)

                found_plates.append((plate, status))

        return render_template('index.html', result=found_plates, file_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
