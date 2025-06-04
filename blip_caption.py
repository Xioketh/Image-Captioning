from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import argparse

# Command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True, help='Path to the input image')
args = parser.parse_args()

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load and preprocess image
image = Image.open(args.image).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Generate caption
with torch.no_grad():
    output = model.generate(**inputs, max_length=50)

# Decode output
caption = processor.decode(output[0], skip_special_tokens=True)
print("🖼️ Caption:", caption)