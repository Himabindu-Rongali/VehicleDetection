import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # For vehicle detection

def detect_vehicle_and_plate(frame):
    results = model(frame)
    vehicles = []
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            x1, y1, x2, y2 = map(int, box)
            vehicles.append(frame[y1:y2, x1:x2])
    return vehicles
