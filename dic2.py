import easyocr
import cv2
from spellchecker import SpellChecker

reader = easyocr.Reader(['en'], gpu=False)


image_path = "pic.jpg"   
img = cv2.imread(image_path)

results = reader.readtext(img, detail=0, paragraph=False)

print("\n======= RAW OCR LINES =======")
for line in results:
    print(line)


spell = SpellChecker()
custom_dict = {
    "thiazide", "chlorothiazide", "nephron", "edema", "hypertension",
    "heart", "failure", "liver", "disease", "sodium", "chloride",
    "diuretic", "blood", "pressure", "volume", "excretion"
}

def correct_word(w):
    w_clean = "".join([c for c in w if c.isalpha()])  
    

    if w_clean.lower() in custom_dict:
        return w_clean

    if w_clean.lower() in spell:
        return w_clean

    corrected = spell.correction(w_clean)
    return corrected if corrected else w_clean


final_lines = []

for line in results:
    words = line.split()
    corrected = " ".join(correct_word(w) for w in words)
    final_lines.append(corrected)


print("\n======= FINAL OCR (LINE BY LINE) =======")
for line in final_lines:
    print(line)


with open("final_line_output.txt", "w", encoding="utf-8") as f:
    for line in final_lines:
        f.write(line + "\n")

print("\nSaved to final_line_output.txt")
