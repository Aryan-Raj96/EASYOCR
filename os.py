import cv2
import easyocr
from spellchecker import SpellChecker

# ============================================================
# 1. LOAD OCR
# ============================================================
reader = easyocr.Reader(['en'], gpu=False)

# ============================================================
# 2. LOAD & PREPROCESS IMAGE
# ============================================================
img = cv2.imread("os.jpg")

# Convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Noise remove
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# Adaptive threshold (BEST for handwriting)
gray = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31, 5
)

# Save processed image (optional)
cv2.imwrite("processed.jpg", gray)

# ============================================================
# 3. OCR (Line by Line)
# ============================================================
results = reader.readtext(
    gray,
    detail=1,
    paragraph=False,
    text_threshold=0.4,
    link_threshold=0.4,
    low_text=0.3
)


results = sorted(results, key=lambda x: x[0][0][1])

spell = SpellChecker()

# Big OS Subject Dictionary (expand anytime)
custom_dict = {
    "Multiprogramming", "Operating", "System", "CPU", "I/O", "Job",
    "Process", "Perform", "Scheduled", "Utilization", "Increase",
    "Decrease", "Throughput", "Linux", "Windows", "Unix",
    "Multiprocessor", "Advantages", "Reliability", "Support",
    "Fault", "Tolerance", "Idle", "Operation", "Example"
}

def correct_word(word):
    w = word.strip()

    # Already valid
    if w.lower() in [c.lower() for c in custom_dict]:
        return w

    # Spell correction
    corrected = spell.correction(w)
    if corrected:
        return corrected

    return w

# ============================================================
# 5. RECONSTRUCT EXACT LINES
# ============================================================
final_lines = []

for box, text, prob in results:
    words = text.split()
    cleaned = " ".join(correct_word(w) for w in words)
    final_lines.append(cleaned)

# ============================================================
# 6. PRINT FINAL OUTPUT (EXACT INPUT STYLE)
# ============================================================
print("\n====== FINAL CLEAN LINE-BY-LINE OUTPUT ======\n")
for line in final_lines:
    print(line)

# Save to a text file
with open("Final_OCR_Output.txt", "w", encoding="utf-8") as f:
    for line in final_lines:
        f.write(line + "\n")

print("\nSaved → Final_OCR_Output.txt")
