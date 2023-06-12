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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM

""" Preprocesssing """
# C:\Yuva Karthik\Yuva Karthik\Datasets\Mask_Dataset
data_path = r"C:\Yuva Karthik\Yuva Karthik\Datasets\Mask_Dataset"
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
    folder_path = os.path.join(data_path, category) # since there are 2 categories (mask and no mask)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name) # the path for each image
        img = cv2.imread(img_path)

        try:
            #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Gray Scale
            imgResize = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) # resize it to 100 x 100
            data.append(imgResize)
            target.append(label_dict[category])

        # We used try/except so that, if there are any errors, it will ignore and pass on to the next image
        except Exception as e: 
            print('Exception: ', e) 

# Normalize

data = np.array(data, dtype = np.float64) / 255.0 # normalize
print(data[0:5])
#data = np.reshape(data, (data.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE))

target = np.array(target, dtype = np.float64)
print(target)
new_target = np_utils.to_categorical(target, 2) # have it in a categorical manner
print(new_target)
train_x, test_x, train_y, test_y = train_test_split(data, new_target, test_size = 0.2, random_state = 42)

print(train_x.shape)
print(train_y.shape)

"""
# Pretrained model mobile netv which has a visual recognition network
baseModel = MobileNetV2(weights = "imagenet", include_top=False, input_tensor = Input(shape = (224, 224, 3))) 

headModel = baseModel.output
headModel = Conv2D(200, (3, 3), activation = 'relu', input_shape = (224, 224, 3))(headModel)
headModel = MaxPooling2D(pool_size = (2, 2))(headModel)

headModel = Conv2D(100, (3, 3), activation = 'relu')(headModel)
headModel = MaxPooling2D(pool_size = (2, 2))(headModel)

headModel = Conv2D(50, (3, 3), activation = 'relu')(headModel)
headModel = MaxPooling2D(pool_size = (2, 2))(headModel)

headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(50, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = "softmax")(headModel)

model = Model(inputs = baseModel.input, outputs = headModel)

# WORKED NO.1
baseModel = MobileNetV2(weights = "imagenet", include_top=False, input_tensor = Input(shape = (224, 224, 3))) 

for layer in baseModel.layers:
    layer.trainable = False

model = Sequential()
model.add(baseModel)
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

# WORKED NO.2

model = Sequential()
model.add(Conv2D(96, (3,3), activation = 'relu', input_shape = (224, 224, 3), padding = 'valid'))
model.add(AveragePooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Conv2D(512, (3,3), activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(128,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

# WORKED NO.3

model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (224, 224, 3), padding = 'valid'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

# WORKED - Mask abitDown
model = Sequential()
model.add(Conv2D(96, (3,3), activation = 'relu', input_shape = (224, 224, 3), padding = 'valid'))
model.add(AveragePooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Conv2D(512, (3,3), activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(128,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))

model = Sequential()
model.add(Conv2D(36, (3,3), activation = 'relu', input_shape = (224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(36, (3,3), activation = 'relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.25))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(64,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))

baseModel = MobileNetV2(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))
for layer in baseModel.layers:
    layer.trainable = False

model = Sequential()
model.add(baseModel)
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(2,activation = 'softmax'))
"""
print('bla')
model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (224, 224, 3), padding = 'valid'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2,activation='softmax'))


INIT_LR = 0.0001
EPOCHS = 20
BS = 32

train_datagen = ImageDataGenerator(
                    rotation_range = 10,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True,
                    fill_mode = 'nearest')


Estop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1, min_delta = 0.01)
opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

model.summary()

hist = model.fit(
    train_datagen.flow(train_x, train_y, batch_size = BS),
    steps_per_epoch = train_x.shape[0] // BS,
    validation_data = (test_x, test_y),
    validation_steps = test_x.shape[0] // BS,
    callbacks = [Estop],
    epochs = EPOCHS
)

test_pred = model.predict(test_x, batch_size = BS)
plt.imshow(test_x[1])
plt.show()
test_pred = np.argmax(test_pred, 1)
print(test_pred[1])

# Classification Report 
print(classification_report(test_y.argmax(1), test_pred, target_names = ['with_mask', 'without_mask']))


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


model.save('My_MobileNet_tensorflow_mask_detector', save_format = 'h5')
