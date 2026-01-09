import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "line.jpg"

# ================= STEP 0: LOAD IMAGE =================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Image not loaded")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ================= STEP 1: DESKEW (ROBUST) =================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Inverted binary (text = white)
_, bw = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Morphology to strengthen text (accuracy boost)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

# Compute skew angle
coords = np.column_stack(np.where(bw > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

(h, w) = gray.shape
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, angle, 1.0)
deskewed = cv2.warpAffine(
    img, M, (w, h),
    flags=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REPLICATE
)

deskewed_rgb = cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB)

# ================= STEP 2: GRAYSCALE + CLEAN B/W =================
gray_d = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

bw_d = cv2.adaptiveThreshold(
    gray_d, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31, 15
)

# Strong morphology for clean text blocks
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
bw_d = cv2.morphologyEx(bw_d, cv2.MORPH_CLOSE, kernel, iterations=2)

# ================= STEP 3: CONTENT-BASED CROP (NO GUESS) =================
row_sum = np.sum(bw_d, axis=1)
col_sum = np.sum(bw_d, axis=0)

rows = np.where(row_sum > 0)[0]
cols = np.where(col_sum > 0)[0]

if len(rows) == 0 or len(cols) == 0:
    raise ValueError("No text detected for cropping")

y1, y2 = rows[0], rows[-1]
x1, x2 = cols[0], cols[-1]

final_crop = deskewed_rgb[y1:y2, x1:x2]

# ================= FINAL OUTPUT =================
print("Skew angle corrected:", round(angle, 2))
print("Original shape:", img_rgb.shape)
print("Final cropped shape:", final_crop.shape)

plt.figure(figsize=(8, 3))
plt.imshow(final_crop)
plt.title("FINAL OUTPUT: Deskewed + B/W + Text Cropped")
plt.axis("off")
