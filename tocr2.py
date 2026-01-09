from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ALWAYS use same model + processor
model_name = "microsoft/trocr-base-handwritten"

processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# load image
image = Image.open("abc.png").convert("RGB")

# preprocess
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generate text
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\nExtracted Text:\n", text)
