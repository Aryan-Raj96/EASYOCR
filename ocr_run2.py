import cv2
import easyocr

reader=easyocr.Reader(['en'],gpu=False)

img= cv2.imread('abc.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,th=cv2.threshold(gray,150,255,cv2.THRESH_BINARY)

#OCR
result=reader.readtext(th)

print("Detected text:-")
for(bbox,text,conf) in result:
    print(text)