import cv2
import numpy as np
import easyocr
from spellchecker import SpellChecker

# Load image
img = cv2.imread("notes.jpg")

# PREPROCESSING
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 9, 75, 75)
th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY,11,2)

# OCR
reader = easyocr.Reader(['en'])
result = reader.readtext(th)

# Spellchecker
spell = SpellChecker()

paragraph = []
current_line_y = None
line_text = []

for (bbox, text, prob) in result:


    corrected_words = []
    for word in text.split():
        corrected = spell.correction(word)
        if corrected is None:    
            corrected = word
        corrected_words.append(corrected)

    text = " ".join(corrected_words)
    

    # y-position for grouping lines
    y = bbox[0][1]

    if current_line_y is None:
        current_line_y = y

    if abs(y - current_line_y) > 20:  # New line detected
        paragraph.append(" ".join(line_text))
        line_text = []
        current_line_y = y

    line_text.append(text)

# add last line
if line_text:
    paragraph.append(" ".join(line_text))

# final paragraph
final_output = "\n".join(paragraph)

print("\n===== FINAL CLEAN PARAGRAPH =====\n")
print(final_output)
