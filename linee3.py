import cv2
import easyocr
import os


# PATHS
IMAGE_PATH = "Line.jpg"
OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "result.txt")

# Load image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# EasyOCR (NO preprocessing)
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(img)

final_text = []

for (bbox, text, conf) in results:
    if conf > 0.30:
        final_text.append(text)

# Save + Print
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
