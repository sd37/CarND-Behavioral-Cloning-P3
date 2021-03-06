import argparse
import base64
from datetime import datetime
import os
import shutil

import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

max_speed_limit = -1
min_speed_limit = -1

def preprocess_image(image_array):
    # convert image_array to BGR format
    ch, row, col = 3, 40, 80
    image_array = cv2.resize(image_array, (col,row))
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_array

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]

        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        print("Image details:", image)

        image_array = np.asarray(image)  
        image_array = preprocess_image(image_array)

        print("predicting steering_angle ..")
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        # control speed of vehicle
        min_speed = min_speed_limit
        max_speed = max_speed_limit
        if float(speed) < min_speed:
            throttle = 1.0
        elif float(speed) > max_speed:
            throttle = -1.0
        else:
            throttle = 0.05


        if abs(float(steering_angle)) > 0.25:
            print("SHARP TURN!!")
            steering_angle = steering_angle * 15

        print(steering_angle, throttle, speed)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )

    parser.add_argument(
            'max_speed_limit',
            type=int,
            nargs='?',
            default=15,
            help='Max speed limit of your track. default = 15')

    parser.add_argument(
            'min_speed_limit',
            type=int,
            nargs='?',
            default=10,
            help='min speed limit of your track. default = 10')

    args = parser.parse_args()
    model = load_model(args.model)

    min_speed_limit = args.min_speed_limit
    max_speed_limit = args.max_speed_limit

    print("min_speed_limit:", min_speed_limit)
    print("max_speed_limit:", max_speed_limit)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
