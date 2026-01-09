import easyocr
import cv2
from spellchecker import SpellChecker

# -----------------------------
# 1. LOAD OCR READER
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# 2. LOAD IMAGE
# -----------------------------
image_path = "pic.jpg"  # <-- same folder me image rakho
img = cv2.imread(image_path)

# -----------------------------
# 3. OCR EXTRACTION
# -----------------------------
result = reader.readtext(img, detail=0, paragraph=True)

raw_text = " ".join(result)
print("\n======= RAW OCR OUTPUT =======")
print(raw_text)

# -----------------------------
# 4. ENGLISH SPELL CHECKER
# -----------------------------
spell = SpellChecker()

# -----------------------------
# 5. CUSTOM SUBJECT DICTIONARY
# -----------------------------
custom_dict = {
    "thiazide", "chlorothiazide", "nephron", "edema", "hypertension",
    "heart", "failure", "liver", "disease", "sodium", "chloride",
    "diuretic", "blood", "pressure", "volume", "excretion"
}

# -----------------------------
# 6. REAL ENGLISH WORD LIST
# -----------------------------
# SpellChecker ke dictionary auto load ho jati hai
english_words = spell.word_frequency.load_words(list(spell.word_frequency.words()))

# -----------------------------
# 7. FINAL CORRECTION LOGIC
# -----------------------------
def correct_word(w):

    w_clean = "".join([c for c in w if c.isalpha()])  # remove punctuation

    if w_clean.lower() in custom_dict:     # important medical words → don't change
        return w_clean

    if w_clean.lower() in spell:           # valid English word
        return w_clean

    # Otherwise correct the spelling
    corrected = spell.correction(w_clean)
    return corrected if corrected else w_clean


# -----------------------------
# 8. APPLY CORRECTIONS
# -----------------------------
words = raw_text.split()
corrected_words = [correct_word(w) for w in words]

final_text = " ".join(corrected_words)

print("\n======= FINAL CLEANED OCR TEXT =======")
print(final_text)

# -----------------------------
# 9. SAVE OUTPUT
# -----------------------------
with open("final_output.txt", "w", encoding="utf-8") as f:
    f.write(final_text)

print("\nSaved to final_output.txt")
