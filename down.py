import easyocr

reader= easyocr.Reader(['en'],gpu=False)

image_path='download.jpg'

result=reader.readtext(image_path)

print("Detected text:-")
for box,text,conf in result:
    print(text)