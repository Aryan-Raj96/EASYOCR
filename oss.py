import easyocr
from spellchecker import SpellChecker
import cv2

reader = easyocr.Reader(['en'], gpu=False)

img = cv2.imread("os.jpg")

results = reader.readtext(img, detail=1, paragraph=False)


results = sorted(results, key=lambda x: x[0][0][1])

spell = SpellChecker()

custom_dict = {
    "Multiprogramming", "Operating", "System", "CPU", "I/O",
    "Utilization", "Throughput", "Linux", "Unix", "Windows",
    "Multiprocessor", "Fault", "Tolerance", "Reliability"
}

def correct_word(word):
    clean = "".join([c for c in word if c.isalnum()])
    if clean.lower() in [c.lower() for c in custom_dict]:
        return clean
    corrected = spell.correction(clean)
    return corrected if corrected else clean

final_lines = []

for box, text, prob in results:
    # detect approximate spaces by box width
    (p1, p2, p3, p4) = box
    line_width = p2[0] - p1[0]

    words = text.split()
    cleaned = " ".join(correct_word(w) for w in words)

    final_lines.append(cleaned)

print("\n====== FINAL CLEAN LINE BY LINE OUTPUT ======\n")
for line in final_lines:
    print(line)
with open("OS_OCR_output.txt", "w", encoding="utf-8") as f:
    for line in final_lines:
        f.write(line + "\n")

print("\nSaved → OS_OCR_output.txt")
