import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cloudinary
import cloudinary.uploader
from io import BytesIO

cloudinary.config(
    cloud_name=st.secrets["CLOUDINARY"]["cloud_name"],
    api_key=st.secrets["CLOUDINARY"]["api_key"],
    api_secret=st.secrets["CLOUDINARY"]["api_secret"]
)


def compress_image(image_file, max_size=(1024, 1024)):
    image = Image.open(image_file)
    image.thumbnail(max_size)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return buffer

def upload_to_cloudinary(file, filename):
    result = cloudinary.uploader.upload(file, public_id=filename, resource_type="image")
    return result["secure_url"]

# Load BLIP model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

st.title("🖼️ Image Captioning with BLIP")
st.write("Upload an image to generate a caption using the BLIP Transformer model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    compressed_image = compress_image(uploaded_file)
    cloudinary_url = upload_to_cloudinary(compressed_image, uploaded_file.name)

    processor, model = load_model()

    image = Image.open(uploaded_file).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)

    caption = processor.decode(output[0], skip_special_tokens=True)
    st.markdown("### 📝 Generated Caption:")
    st.success(caption)
