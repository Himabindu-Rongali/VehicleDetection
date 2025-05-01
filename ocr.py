import easyocr
import cv2

reader = easyocr.Reader(['en'])  # Add languages like 'hi' for Hindi

def read_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    plate_numbers = [res[1] for res in result if len(res[1]) > 5]
    return plate_numbers
