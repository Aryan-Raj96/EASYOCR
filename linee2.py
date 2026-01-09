import cv2
import easyocr
import os

# -----------------------------
# PATHS
# -----------------------------
IMAGE_PATH = "Line.jpg"
OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "result.txt")

# -----------------------------
# Load image
# -----------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Preprocessing (light)
# -----------------------------
blur = cv2.GaussianBlur(gray, (3, 3), 0)
_, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# -----------------------------
# EasyOCR (Direct)
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(thresh)

final_text = []

for (bbox, text, conf) in results:
    if conf > 0.35:
        final_text.append(text)

# -----------------------------
# Save + Print
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for line in final_text:
        f.write(line + "\n")

print("\n========== OCR OUTPUT ==========")
if final_text:
    for line in final_text:
        print(line)
else:
    print("⚠️ No text detected")
print("================================")

print("\nOCR Completed ✅")
print("Output saved to:", OUTPUT_PATH)
