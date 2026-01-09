from paddleocr import PaddleOCR

# OCR object initialize
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # English handwriting support

# OCR run
result = ocr.ocr("abc.png", cls=True)

# Output print
for line in result:
    print(line[1][0])
