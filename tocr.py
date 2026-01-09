# trocr_inference.py
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# 1️⃣ Load pretrained Trocr model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# 2️⃣ Load your image
image_path = "abc.png"  # <-- image ka naam ya path
image = Image.open(image_path).convert("RGB")

# 3️⃣ Preprocess image
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# 4️⃣ Generate text
generated_ids = model.generate(pixel_values)

# 5️⃣ Decode to readable text
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Extracted Text:")
print(text)
