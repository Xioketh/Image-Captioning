# Image Captioning Project

## Introduction

This project generates descriptive captions for images using deep learning techniques. Initially, we implemented an LSTM-based approach, but have since transitioned to using the more advanced BLIP (Bootstrapped Language-Image Pre-training) model, which provides significantly better results with less complexity.

## Old Process (LSTM-based)

Our initial approach used a traditional encoder-decoder architecture with LSTM networks:

1. **Feature Extraction**: Used Xception network to extract image features
2. **Tokenizer Preparation**: Created a text tokenizer from the training captions
3. **Model Training**: Trained an LSTM network to decode features into captions
4. **Caption Generation**: Generated captions for new images using the trained model

**Dependencies**:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib (for visualization)
- tqdm (for progress bars)

**Dataset**: Flickr8k dataset (8,000 images with 5 captions each)

**Limitations**:
- Long training time required
- Moderate accuracy with simple captions
- Complex pipeline with separate feature extraction
- Limited vocabulary from training data
- Challenges with novel image compositions

## New Process (BLIP-based)

The BLIP model provides a more sophisticated and efficient solution:

- End-to-end vision-language pre-trained model
- No separate feature extraction needed
- Generates more natural and accurate captions
- Pre-trained on large datasets with broad vocabulary

**Dependencies**:
- PyTorch
- Hugging Face Transformers
- PIL (Python Imaging Library)

**Steps to run BLIP code**:
1. Install required packages
2. Load the pre-trained BLIP model
3. Process your input image
4. Generate caption with a single forward pass

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img = Image.open("your_image.jpg").convert('RGB')
inputs = processor(img, return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))