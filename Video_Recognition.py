import numpy as np 
import cv2
import os
import torch
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf

#model = tf.keras.models.load_model('My_MobileNet_tensorflow_mask_detector.h5')
#model = tf.keras.models.load_model('tensorflow_mask_detector2.h5')
model = tf.keras.models.load_model(r"C:\Yuva Karthik\Yuva Karthik\python_projects\12k_Mask.h5")

face_clsfr = cv2.CascadeClassifier(r"C:\Yuva Karthik\Yuva Karthik\Datasets\frontalFace10\haarcascade_frontalface_default.xml")


source = cv2.VideoCapture(0)
#URL = 'http://10.0.0.186:8080/video'
#source.open(URL)

labels_dict = {1:'MASK', 0:'NO MASK'}
color_dict = {1 : (0, 255, 0), 0 : (0, 0, 255)}

while(True):

    ret,img = source.read()
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(img,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img = img[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (224,224))
        normalized = np.array(resized, dtype = np.float64) / 255.0
        reshaped = np.reshape(normalized,(1,224,224,3))
        result = model.predict(reshaped)
        print(result)

        label = np.argmax(result, 1)[0]

        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label.item()], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        print(label.item())
        
    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
source.release()
