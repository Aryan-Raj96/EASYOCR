import easyocr

reader=easyocr.Reader(['en'])

image_path='abc.png'

result=reader.readtext(image_path)

print("DETECTED TEXT:")
for box,text,conf in result:
    print(text)
