from flask import Flask, render_template, request, send_file
import os
import cv2
from ultralytics import YOLO
from ocr import read_plate
import sqlite3
from datetime import datetime
import csv
from fpdf import FPDF

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

@app.route('/download/<file_format>')
def download_logs(file_format):
    conn = sqlite3.connect('vehicles.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, plate, status FROM logs ORDER BY timestamp DESC")
    logs = c.fetchall()
    conn.close()

    if file_format == 'csv':
        filename = "logs.csv"
        filepath = os.path.join('static', filename)

        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'License Plate', 'Status'])
            formatted_logs = [(datetime.strptime(log[0], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S"), log[1], log[2]) for log in logs]
            writer.writerows(formatted_logs)

        return send_file(filepath, as_attachment=True)

    elif file_format == 'pdf':
        filename = "logs.pdf"
        filepath = os.path.join('static', filename)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Vehicle Detection Logs", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(60, 10, 'Timestamp', border=1, align='C')
        pdf.cell(60, 10, 'License Plate', border=1, align='C')
        pdf.cell(60, 10, 'Status', border=1, align='C')
        pdf.ln()

        for log in logs:
            pdf.cell(60, 10, log[0], border=1)
            pdf.cell(60, 10, log[1], border=1)
            pdf.cell(60, 10, log[2], border=1)
            pdf.ln()

        pdf.output(filepath)
        return send_file(filepath, as_attachment=True)

    return "Invalid format selected", 400

@app.route('/logs')
def view_logs():
    conn = sqlite3.connect('vehicles.db')
    c = conn.cursor()
    page = request.args.get('page', 1, type=int)
    logs_per_page = 15
    offset = (page - 1) * logs_per_page
    c.execute("SELECT timestamp, plate, status FROM logs ORDER BY timestamp DESC LIMIT ? OFFSET ?", (logs_per_page, offset))
    logs = c.fetchall()
    c.execute("SELECT COUNT(*) FROM logs")
    total_logs = c.fetchone()[0]
    conn.close()
    total_pages = (total_logs + logs_per_page - 1) // logs_per_page
    return render_template('logs.html', logs=logs, page=page, total_pages=total_pages)

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
                        break

            output_path = os.path.join(UPLOAD_FOLDER, "processed_" + filename)
            cv2.imwrite(output_path, img)
            file_path = '/' + output_path

    return render_template('index.html', result=result, file_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
