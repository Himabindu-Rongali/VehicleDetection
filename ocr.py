from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def read_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = ocr.ocr(gray, cls=True)
    plate_numbers = []

    if result:
        for line in result:
            if line is not None:
                for box in line:
                    text = box[1][0]
                    if 5 <= len(text) <= 15:
                        plate_numbers.append(text)
    return plate_numbers
