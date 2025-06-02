from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception

from keras.models import load_model

from pickle import load
import numpy as np
from PIL import Image
import argparse
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image")
args = vars(ap.parse_args())
img_path = args['image']

def extract_features(filename, model):
    try:
        image = Image.open(filename).convert('RGB')  # Convert to RGB
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    image = image.resize((299, 299))
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

def genrate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        # pred = model.predict([photo, sequence], verbose=0)
        # pred = np.argmax(pred)
        pred = model.predict([photo, sequence], verbose=0)
        pred = sample(pred, temperature=0.8)  # tweak as needed

        word = word_for_id(pred, tokenizer)
        print(f"Step {i}: predicted word ->", word)  # Debug
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


def define_model(vocab_size, max_length):
    # CNN model from 2048 nodes to 256 nodes
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # LSTM sequence model
    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decorder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decorder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds[0], 1)
    return np.argmax(probas)


max_length =32
tokenizer = load(open('tokenizer.p', 'rb'))
vocab_size = len(tokenizer.word_index) + 1

model =  define_model(vocab_size, max_length)
model.load_weights("models/model_34.h5")
xception_model =  Xception(include_top=False, pooling='avg')

photo =  extract_features(img_path, xception_model)
img=Image.open(img_path)

print("Tokenizer vocab size:", len(tokenizer.word_index))
print("Max length used:", max_length)
print("Feature shape:", photo.shape)

print("--------------------------------------------")
print("--------------------------------------------")
print("--------------------------------------------")

# print(tokenizer.index_word[1:10])



description = genrate_desc(model, tokenizer, photo, max_length)
print(description)