import easyocr
import os
import cv2
from spellchecker import SpellChecker
image_folder = "images"

for filename in os.listdir(image_folder):
    if filename.lower() == "abc.png":  # sirf abc.png process hoga
        image_path = os.path.join(image_folder, filename)
        
        # Preprocessing
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_bin = cv2.medianBlur(img_bin, 3)
        temp_path = "temp.png"
        cv2.imwrite(temp_path, img_bin)
        
        # OCR
        result = reader.readtext(temp_path, detail=0)
        raw_text = " ".join(result)
        print(f"\n--- Raw OCR Output ({filename}) ---")
        print(raw_text)

        # Dictionary correction
        corrected_words = []
        for word in raw_text.split():
            corrected_word = spell.correction(word)
            corrected_words.append(corrected_word)
        
        corrected_text = " ".join(corrected_words)
        print(f"\n--- Corrected Text ({filename}) ---")
        print(corrected_text)
