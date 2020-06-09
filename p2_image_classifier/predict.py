## imports
import argparse
import numpy as np
import json
import time

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from PIL import Image

## Arguments 

parser = argparse.ArgumentParser(description = "Image_classification")

parser.add_argument("image_path", action="store")
parser.add_argument("model",action="store", default = 'model.h5')
parser.add_argument("--top_k", action="store", type=int, default = 5)
parser.add_argument("--category_names", action="store", default = "label_map.json")
args = parser.parse_args()
print(args)

with open('label_map.json', 'r') as f:
    class_names = json.load(f)
    
## Load_model
reload_model = tf.keras.models.load_model(args.model,custom_objects={'KerasLayer': hub.KerasLayer})


## Image predection
def process_image(image):
    image = tf.convert_to_tensor(image,dtype = tf.float16)
    image = tf.image.resize(image,(224, 224))
    image /= 255
    image = image.numpy()
    return image

def predict(image, model, top_k):
    img = np.asarray(Image.open(image))
    
    processed_img = process_image(img)
    processed_img = np.expand_dims(processed_img,axis = 0)
    
    predection = reload_model.predict(processed_img)
    top_prop, top_classes = tf.math.top_k(predection, top_k)


    prop_list = top_prop[0].numpy().tolist()
    classes_list = top_classes[0].numpy().tolist()

    return prop_list, classes_list

## Main 

prop, classes = predict(args.image_path,args.model,args.top_k)
print("Top probabilities", prop)
print("Top class labels", classes)
    