import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

gesture = ['Paper','Rock','Scissors']

cam = cv2.VideoCapture(0)
# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
while True:
    # Read the frame
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fc = cv2.resize(frame, (224,224))
    inp = np.reshape(fc,(1,224,224,3)).astype(np.float32)

    image_array = np.asarray(inp)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)

    gest = gesture[np.argmax(prediction)]

    # display the resized image
    cv2.imshow('frame', frame)
    #print(prediction)
    print(gest)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
