import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import enchant

def show(title, img):
    plt.figure(figsize=(6,3))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

# Load
img = cv2.imread("pointer.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
show("1. Original Image", rgb)

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show("2. Grayscale", gray)

# Black & White
blur = cv2.GaussianBlur(gray, (5,5), 0)
_, bw = cv2.threshold(blur, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show("3. Black & White", bw)

# Deskew
coords = np.column_stack(np.where(bw > 0))
angle = cv2.minAreaRect(coords)[-1]
angle = -(90 + angle) if angle < -45 else -angle

(h, w) = bw.shape
M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
deskew = cv2.warpAffine(bw, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)
show("4. Deskewed", deskew)

# Crop
cnts, _ = cv2.findContours(deskew, cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(np.vstack(cnts))
crop = deskew[y:y+h, x:x+w]
show("5. Cropped Text", crop)

# OCR
reader = easyocr.Reader(['en'], gpu=False)
text = reader.readtext(crop, detail=0)

# Dictionary filter
d = enchant.Dict("en_US")
valid = [w for w in text if w.isalpha() and d.check(w.lower())]

print("FINAL OCR TEXT:")
print(" ".join(text))

print("\nDICTIONARY VERIFIED:")
print(" ".join(valid))
