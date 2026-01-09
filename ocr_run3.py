import cv2
import easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

img = cv2.imread('abc.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# adaptive threshold for handwriting
th = cv2.adaptiveThreshold(gray,255,
                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY,15,8)

result = reader.readtext(th, detail=1)

# sort by Y then X coordinate
result = sorted(result, key=lambda r: (r[0][0][1], r[0][0][0]))

final_text = ""
for bbox, text, conf in result:
    final_text += text + " "

print("\nFinal Ordered Text:\n")
print(final_text)
