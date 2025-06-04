import streamlit as st
from PIL import Image
import numpy as np
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import tensorflow as tf


def upload_to_cloudinary(file, filename):
    result = cloudinary.uploader.upload(file, public_id=filename, resource_type="image")
    return result["secure_url"]

# ---------------- Utility Functions ---------------- #
def extract_features(image, model):
    image = image.resize((299, 299)).convert('RGB')
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds[0], 1)
    return np.argmax(probas)

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = sample(yhat, temperature=0.8)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text.replace('start', '').replace('end', '').strip()

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

# ------------------- Streamlit UI ------------------- #
st.title("🖼️ Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load tokenizer and model only after uploading image
    tokenizer = load(open('tokenizer.p', 'rb'))
    max_length = 32
    vocab_size = len(tokenizer.word_index) + 1

    # Load models
    xception_model = Xception(include_top=False, pooling='avg')
    caption_model = define_model(vocab_size, max_length)
    caption_model.load_weights("models/model_99.h5")

    # Extract features and generate caption
    photo = extract_features(image, xception_model)
    caption = generate_desc(caption_model, tokenizer, photo, max_length)

    st.markdown("### 📌 Generated Caption:")
    st.success(caption)
