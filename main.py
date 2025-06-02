from pydoc import describe

import numpy as np
import string
import pandas as pd
import os
from pickle import dump, load
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, get_file
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Flatten, Dropout, Bidirectional
from PIL import Image

# from tqdm.notebook import tqdm
from tqdm import tqdm

tqdm.pandas()

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))




def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def all_img_captions(filename):
    file = load_doc(filename)
    captions =  file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        img,caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption.replace("-","")
            desc = img_caption.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if (len(word)) > 1]
            desc = [word for word in desc if(word.isalpha())]

            img_caption = ' '.join(desc)
            captions[img][i] = img_caption
    return captions

def text_vocabulary(descriptions):
    vocab= set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab

def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key+'\t'+desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

dataset_text = "Flickr8k_text"
dataset_images = "Flicker8k_Dataset"

filename = dataset_text+"/Flickr8k.token.txt"
descriptions  = all_img_captions(filename)
print("length of description:: ",len(descriptions))

clean_descriptions = cleaning_text(descriptions)
vocabulary = text_vocabulary(clean_descriptions)
print("vocabulary size: ",len(vocabulary))

save_descriptions(clean_descriptions, "descriptions.txt")

# end=-------------------------

features = load(open("features.p", 'rb'))


def load_photos(filename):
    file = load_doc(filename)
    photos = file.strip().split('\n')

    missing = 0
    photos_present = []

    for photo in photos:
        full_path = os.path.join(dataset_images, photo)
        if os.path.exists(full_path):
            photos_present.append(photo)
        else:
            missing += 1

    print(f"Found {len(photos_present)} images, {missing} missing.")
    return photos_present


def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.split('\n'):
        words = line.split()
        if len(words) < 1:
            continue

        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> '+" ".join(image_caption)+' </end>'
            descriptions[image].append(desc)

    return descriptions


def load_features(photos):
    all_features = load(open("features.p", 'rb'))
    features = {k: all_features[k] for k in photos if k in all_features}

    missing = [k for k in photos if k not in all_features]
    if missing:
        print(f"Warning: {len(missing)} images missing in features.p")
        print("Some missing examples:", missing[:5])

    print(f"Loaded {len(features)} features out of {len(photos)} requested.")
    return features


filename = dataset_text + "/"+"Flickr_8k.trainImages.txt"

train_imgs = load_photos(filename)
print("Train image filenames:", len(train_imgs))
print(train_imgs[:5])
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)

def dict_to_list(descriptions):
    all_desc =[]
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

tokenizer = create_tokenizer(train_descriptions)

dump(tokenizer, open("tokenizer.p", 'wb'))

vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size: ",vocab_size)

def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

print("Total keys in train_descriptions:", len(train_descriptions))
max_length = max_length(train_descriptions)
print("Max Length: ",max_length)


def data_generator(descriptions, features, tokenizer, max_length):
    def generator():
        while True:
            for key, description_list in descriptions.items():
                feature = features[key][0]
                input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list,
                                                                            feature)
                for i in range(len(input_image)):
                    yield {'input_1': input_image[i], 'input_2': input_sequence[i]}, output_word[i]

    # Define the output signature for the generator
    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )

    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    return dataset.batch(32)

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)


dataset = data_generator(train_descriptions, features, tokenizer, max_length)
print("------------------im here1--------------")
for (a, b) in dataset.take(1):
    print(a['input_1'].shape, a['input_2'].shape, b.shape)
    break

print("------------------im here1.0--------------")

from keras.utils import plot_model

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

print("------------------im here2--------------")

model = define_model(vocab_size, max_length)
epochs =100
steps_per_epoch = 10
print("------------------im here3--------------")


os.mkdir('models')
print("1111111111111111111111")

with tf.device('/GPU:0'):
    for i in range(epochs):
        dataset = data_generator(train_descriptions, train_features, tokenizer, max_length)
        model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
        model.save('models/model_'+str(i)+'.h5')
