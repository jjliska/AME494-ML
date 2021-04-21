import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

cam = cv2.VideoCapture(0)
# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
while True:
    # Read the frame
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    ret, frame = cam.read()

    fc = cv2.resize(frame, (224,224))
    inp = np.reshape(fc,(1,224,224,3)).astype(np.float32)
    image_array = np.asarray(inp)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)

    # display the resized image
    cv2.imshow('frame', frame)

    print(prediction)
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    #cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
