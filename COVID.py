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

data_path = r"C:\Yuva Karthik\Yuva Karthik\Datasets\covid-chestxray-dataset-master"
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))

print(label_dict)
print(categories)
print(labels)

IMAGE_SIZE = 224
data = []
target = []

for category in categories:
    folder_path = os.path.join(data_path, category) 
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            imgResize = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) 
            data.append(imgResize)
            target.append(label_dict[category])

        except Exception as e:
            print('Exception: ', e) 

# Normalize
print(data[0])
data = np.array(data, dtype = np.float64) / 255.0 
print(data.shape)

target = np.array(target, dtype = np.float64)
new_target = np_utils.to_categorical(target, 2) 

train_x, test_x, train_y, test_y = train_test_split(data, new_target, test_size = 0.2, random_state = 42)

print(train_x.shape)
print(train_y.shape)

# CNN

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

# Compile
LR = 1e-4
EPOCHS = 10
BS = 32

train_datagen = ImageDataGenerator(
                    rotation_range = 10,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True, 
                    fill_mode = 'nearest')

Estop = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1, min_delta = 0.1)
opt = Adam(lr = LR, decay = LR / EPOCHS)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.summary()

hist = model.fit( 
    train_datagen.flow(train_x, train_y, batch_size = BS),
    steps_per_epoch = train_x.shape[0] // BS,
    validation_data = (test_x, test_y),
    validation_steps = test_x.shape[0] // BS,
    #callbacks = [Estop],
    epochs = EPOCHS
)

# Classification Report
test_pred = model.predict(test_x, batch_size = BS)
test_pred = np.argmax(test_pred, 1)
print(classification_report(test_y.argmax(1), test_pred, target_names = ['COVID_xRay', 'NORMAL']))

fig = plt.figure(figsize = (20,20))
plt.subplot(221)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc ="upper left")
plt.show()
    
plt.subplot(222)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model_loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc ="upper left")
plt.show()

model.save('COVID_DETECTOR', save_format = 'h5')


