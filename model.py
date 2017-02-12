### TA_CODE.
import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                correction = 0.2

                c_name = batch_sample[0].strip()
                c_img = cv2.imread(c_name)
                c_img = cv2.resize(c_img, (80,40))
                center_image = c_img
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                l_name = batch_sample[1].strip()
                l_img = cv2.imread(l_name)
                l_img = cv2.resize(l_img, (80,40))
                left_image = l_img
                left_angle = float(batch_sample[3]) + correction
                images.append(left_image)
                angles.append(left_angle)

                r_name = batch_sample[2].strip()
                r_img = cv2.imread(r_name)
                r_img = cv2.resize(r_img, (80,40))
                right_image = r_img
                right_angle = float(batch_sample[3]) - correction
                images.append(right_image)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 40, 80  # Trimmed image format

# TA_CODE ENDS

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.activations import relu, softmax
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row,col,ch)))

# model.add(... finish defining the rest of your model architecture here ...)

model.add(Convolution2D(16, 8, 8, border_mode='same'))
model.add(MaxPooling2D((2, 2), border_mode = 'same'))
model.add(Dropout(0.5))
model.add(Activation('relu'))


model.add(Convolution2D(32, 5, 5, border_mode='same'))
model.add(MaxPooling2D((2, 2), border_mode = 'same'))
model.add(Dropout(0.5))
model.add(Activation('relu'))


model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(MaxPooling2D((2, 2), border_mode = 'same'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Activation('relu'))
model.add(Dense(512))

model.add(Dropout(0.2))

model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
size_validation_samples = len(validation_samples)
model.fit_generator(train_generator, samples_per_epoch= 20480, nb_epoch=5,
            validation_data=validation_generator, 
            nb_val_samples= size_validation_samples)


print("Saving model in .. ")
model.save('model.h5')
