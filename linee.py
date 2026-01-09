import cv2
import easyocr
import numpy as np
import enchant
import os

# PATHS (image same folder me ho)
IMAGE_PATH = "Line.jpg"
OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "result.txt")

# Load image (safe)
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# STEP 1: Preprocessing
blur = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    15,
    3
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# STEP 2: Line Detection
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
detect_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, horizontal_kernel)

contours, _ = cv2.findContours(
    detect_lines,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Sort top to bottom
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

# STEP 3: Word Detection (EasyOCR)
reader = easyocr.Reader(['en'], gpu=False)
dictionary = enchant.Dict("en_US")

final_text = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    line_img = img[y:y+h, x:x+w]

    results = reader.readtext(line_img, detail=0)

    valid_words = []
    for word in results:
        clean_word = ''.join(c for c in word if c.isalpha())
        if clean_word and dictionary.check(clean_word.lower()):
            valid_words.append(clean_word)

    if valid_words:
        final_text.append(" ".join(valid_words))

# STEP 4 & 5: Save + Print Output
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for line in final_text:
        f.write(line + "\n")

print("\n========== OCR OUTPUT ==========")
if final_text:
    for line in final_text:
        print(line)
else:
    print(" No text detected")
print("================================")

print("\nOCR Completed ")
print("Output saved to:", OUTPUT_PATH)
