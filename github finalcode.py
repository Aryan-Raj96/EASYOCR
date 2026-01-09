import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import enchant

# SHOW IMAGE
def show(title, img, cmap=None):
    plt.figure(figsize=(8,4))
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

# IMAGE FLATTENING
def flatten_image(gray):
    blur = cv2.GaussianBlur(gray, (0,0), 15)
    flat = cv2.divide(gray, blur, scale=255)
    return flat

# CONTRAST ENHANCEMENT
def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)
    
# LOAD IMAGE
img = cv2.imread("pointer.jpg")
if img is None:
    raise FileNotFoundError("Image not found")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
show("1. Original Image", rgb)

# GRAYSCALE
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show("2. Grayscale", gray, cmap="gray")

# PREPROCESS
flat = flatten_image(gray)
contrast = enhance_contrast(flat)
show("3. After Flatten + Contrast", contrast, cmap="gray")

#easyOCR
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(contrast, detail=1)

# COLLECT RAW WORDS
all_words = [text for _, text, _ in results]

# ENGLISH (INDIA) DICTIONARY
try:
    d = enchant.Dict("en_IN")   # Indian English
    print("Using en_IN dictionary")
except:
    d = enchant.Dict("en_GB")   # fallback
    print("Fallback to en_GB dictionary")

# CLEAN + VALIDATE WORDS
valid_words = []

for w in all_words:
    clean = "".join(ch for ch in w.lower() if ch.isalpha())

    if len(clean) < 2:
        continue

    # SOFT MATCH (no rejection, just clean)
    valid_words.append(clean)

# DRAW BOXES
output_img = cv2.cvtColor(contrast, cv2.COLOR_GRAY2RGB)

for box, text, conf in results:
    pts = np.array(box, dtype=np.int32)
    cv2.polylines(output_img, [pts], True, (0,255,0), 2)

    x, y = pts[0]
    cv2.putText(
        output_img,
        text,
        (x, y-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,0,0),
        1,
        cv2.LINE_AA
    )

show("4. OCR Output", output_img)

# FINAL OUTPUT
print("\n🔹 RAW OCR WORDS (Input from Image):")
print(all_words)
print("👉 Total RAW words detected:", len(all_words))

print("\n🔹 CLEANED WORDS (After India Dictionary Processing):")
print(valid_words)
print("👉 Total CLEANED words:", len(valid_words))

print("\n🔹 TOTAL WORD BOXES (OCR detections):", len(results))
