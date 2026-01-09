import cv2
import numpy as np
from paddleocr import PaddleOCR
import imutils

# ------------ Preprocessing Filters  -------------

def preprocess(img_path):
    img = cv2.imread(img_path)

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Noise removal (Gaussian Blur)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Thresholding (Black & White)
    thresh = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # 4. Resize (stable shape)
    resized = cv2.resize(thresh, (0, 0), fx=1.2, fy=1.2)

    # 5. Skew Correction
    coords = np.column_stack(np.where(resized > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = resized.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    corrected = cv2.warpAffine(resized, M, (w, h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    cv2.imwrite("processed.png", corrected)
    return "processed.png"


# ------------ OCR Pipeline -------------

ocr = PaddleOCR(lang='en')

img_path = "abc.jpg"
clean_img = preprocess(img_path)

result = ocr.ocr(clean_img, cls=True)

print("\n--------- FINAL OCR TEXT ----------\n")
for line in result:
    for word in line:
        print(word[1][0])
