import cv2
import easyocr

img = cv2.imread("Line.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)

reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(blur, detail=0)

print("\n========== OCR OUTPUT ==========")
print("\n".join(result))
print("================================")
