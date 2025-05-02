from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO
from ocr import read_plate
import sqlite3
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = YOLO("yolov8n.pt")  # Replace with custom model if available

def create_db():
    conn = sqlite3.connect('vehicles.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (timestamp TEXT, plate TEXT, status TEXT)''')
    conn.commit()
    conn.close()

create_db()

def load_authorized_plates():
    if os.path.exists("authorized_vehicles.txt"):
        with open("authorized_vehicles.txt", "r") as f:
            return set(line.strip().upper() for line in f if line.strip())
    return set()

authorized_plates = load_authorized_plates()

def save_log(plate, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect('vehicles.db')
    c = conn.cursor()
    c.execute("INSERT INTO logs (timestamp, plate, status) VALUES (?, ?, ?)",
              (timestamp, plate, status))
    conn.commit()
    conn.close()

    with open("detection_log.txt", "a") as f:
        f.write(f"{timestamp} - Plate: {plate} - Status: {status}\n")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = []
    file_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            file_path = '/' + filepath

            img = cv2.imread(filepath)
            results = model(filepath)
            predictions = results[0].boxes.data.tolist()

            for *box, conf, cls in predictions:
                x1, y1, x2, y2 = map(int, box)
                cropped = img[y1:y2, x1:x2]
                plates = read_plate(cropped)

                for plate_text in plates:
                    plate_text = plate_text.replace(" ", "").upper()
                    if 5 <= len(plate_text) <= 12:
                        status = "Authorized" if plate_text in authorized_plates else "Unauthorized"
                        save_log(plate_text, status)
                        result.append((plate_text, status))

                        color = (0, 255, 0) if status == "Authorized" else (0, 0, 255)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label = f"{plate_text} ({status})"
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        break  # Stop after detecting one valid plate per box

            output_path = os.path.join(UPLOAD_FOLDER, "processed_" + filename)
            cv2.imwrite(output_path, img)
            file_path = '/' + output_path

    return render_template('index.html', result=result, file_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
