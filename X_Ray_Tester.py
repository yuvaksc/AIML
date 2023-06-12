import numpy as np 
import cv2
import os
import torch
from keras.utils import np_utils
import tensorflow as tf
from matplotlib import pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', input_shape = (224, 224, 3)))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))


model = tf.keras.models.load_model('COVID_DETECTOR.h5')

labels_dict = {1:'normal', 0:'corona'}

data_path = r"C:\Yuva Karthik\Yuva Karthik\Datasets\corona_test"
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))

data = []
target = []

for category in categories:
    folder_path = os.path.join(data_path, category) 
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        imgResize = cv2.resize(img, (224,224))
        data.append(imgResize)
        target.append(label_dict[category])

test = np.array(data, dtype = np.float64) / 255.0
test_pred = model.predict(test, batch_size = 32)
test_pred = np.argmax(test_pred, 1) 

target = np.array(target, dtype = np.float64)
new_target = np_utils.to_categorical(target, 2)

print(classification_report(new_target.argmax(1), test_pred, target_names = ['corona', 'normal']))
