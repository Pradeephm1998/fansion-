import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))#loading the data in (224,224)
    img_array = image.img_to_array(img)#converting the img to array (224,224,3)
    expanded_img_array = np.expand_dims(img_array, axis=0)#expanding the dimensions (1,224,224,3) kears lib works on batch images
    preprocessed_img = preprocess_input(expanded_img_array)#preprocess_input img is coverted input  which resnet50 need (3,224,224,1)
    result = model.predict(preprocessed_img).flatten()#2d to  1d coverted
    normalized_result = result / norm(result)# make 0 to 1 in the array

    return normalized_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))
# filenames generating the path of images

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

# feature_list extraction features

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))