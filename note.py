import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr
from spellchecker import SpellChecker

# Spell checker load
spell = SpellChecker()

# LOAD IMAGE
img = cv2.imread("notes.jpg")    # <-- Make sure file exists
if img is None:
    print("❌ Error: Image not found. Check file name/path.")
    exit()

# ---------- PREPROCESSING ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 9, 75, 75)
th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY,11,2)

plt.imshow(th, cmap='gray')
plt.axis('off')
plt.show()

# ---------- OCR ----------
reader = easyocr.Reader(['en'])
ocr_words = reader.readtext(th, detail=0)

corrected_words = []

for word in ocr_words:
    w = word.strip()

    # Ignore short/symbol words
    if len(w) <= 2 or not w.isalpha():
        corrected_words.append(w)
        continue

    # Dictionary correction
    corrected = spell.correction(w)
    
    if corrected:
        corrected_words.append(corrected)
    else:
        corrected_words.append(w)

# Final paragraph
final_text = " ".join(corrected_words)

print("\n📌 Corrected Text:\n")
print(final_text)
