import cv2
import easyocr
import enchant
import os

# ---------- PATHS ----------
IMAGE_PATH = "Line.jpg"
OUT_DIR = "output"
OUT_FILE = os.path.join(OUT_DIR, "result.txt")

# ---------- STEP 1: PREPROCESSING ----------
img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# ---------- STEP 2: LINE DETECTION ----------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

# ---------- STEP 3: WORD DETECTION ----------
reader = easyocr.Reader(['en'], gpu=False)
dictionary = enchant.Dict("en_US")

final_text = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    line_img = img[y:y+h, x:x+w]

    words = reader.readtext(line_img, detail=0)

    # ---------- STEP 4: DICTIONARY MATCH ----------
    valid = []
    for word in words:
        clean = ''.join(c for c in word if c.isalpha())
        if clean and (dictionary.check(clean.lower()) or len(clean) > 2):
            valid.append(clean)

    if valid:
        final_text.append(" ".join(valid))

# ---------- STEP 5: SAVE OUTPUT ----------
os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_FILE, "w") as f:
    f.write("\n".join(final_text))

print("\n========== OCR OUTPUT ==========")
print("\n".join(final_text) if final_text else "No text detected")
print("================================")
print("Saved to:", OUT_FILE)
