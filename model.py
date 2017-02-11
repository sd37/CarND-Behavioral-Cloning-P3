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

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.extend(center_image)
                angles.extend(center_angle)

            # trim image to only see section with road
            X_train = X_train[:,80:,:,:] 
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

# TA_CODE ENDS

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.activations import relu, softmax

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))

# model.add(... finish defining the rest of your model architecture here ...)

model.add(Convolution2D(16, 8, 8, border_mode='same'))
model.add(Activation('relu'))

model.add(Convolution2D(32, 5, 5, border_mode='same'))
model.add(Activation('relu'))

model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dropout(0.5))

model.add(Activation('relu'))
model.add(Dense(512))

model.add(Dropout(.5))

model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
size_validation_samples = len(validation_samples)
model.fit_generator(train_generator, samples_per_epoch= 128,
            validation_data=validation_generator, 
            nb_val_samples= size_validation_samples, nb_epoch=3)


print("Saving model in .. ")
model.save('model.h5')
