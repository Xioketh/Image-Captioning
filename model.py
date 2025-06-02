import numpy as np
import os
from pickle import dump, load
import tensorflow as tf
from keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, get_file
from PIL import Image

# from tqdm.notebook import tqdm
from tqdm import tqdm

tqdm.pandas()

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))


print("-------------------------------------")


dataset_images = "Flicker8k_Dataset"

def download_with_retry(url, filename, max_retry=3):
    for attempt in range(max_retry):
        try:
            return get_file(filename, url)
        except Exception as e:
            if attempt == max_retry-1:
                raise e
            print("download attempt fail")
            time.sleep(3)

weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = download_with_retry(weights_url, 'xception_weights.h5')
model = Xception(include_top=False, pooling='avg', weights=weights_path)

def extract_features(directory):
    features = {}
    valid_images = ['.jpg','.jpeg','.png']
    for img in tqdm(os.listdir(directory)):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            continue
        filename = directory + '/' + img
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image -1.0

        feature = model.predict(image)
        features[img] = feature
    return features

features = extract_features(dataset_images)
dump(features, open('features.p', 'wb'))