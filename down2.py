import easyocr
import cv2
import numpy as np

# -------------------------#
# 1. PRE-PROCESSING STEPS  #
# -------------------------#

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove noise
    noise_removed = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding (clean text for better OCR)
    thresh = cv2.threshold(noise_removed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Save processed temporarily
    processed_path = "processed_image.jpg"
    cv2.imwrite(processed_path, thresh)

    return processed_path


# -------------------------#
# 2. OCR READING           #
# -------------------------#
reader = easyocr.Reader(['en'], gpu=False)

image_path = 'download.jpg'

# Apply preprocessing
processed_img = preprocess_image(image_path)

# Run OCR on processed image
result = reader.readtext(processed_img)

# -------------------------#
# 3. MERGE TEXT PARAGRAPH  #
# -------------------------#

full_text = ""

for box, text, conf in result:
    full_text += text + " "   # spacing for paragraph format

# -------------------------#
# 4. OUTPUT                #
# -------------------------#

print("\nDetected Text (Paragraph Form):\n")
print(full_text.strip())
