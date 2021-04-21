import cv2
import mediapipe as mp
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

gesture = ['Paper','Rock','Scissors']

cam = cv2.VideoCapture(0)
# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    #Setting the screen grab to 224x224
    fc = cv2.resize(image, (224,224))
    #Recoloring image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      #Creating an array from the image
      inp = np.reshape(fc,(1,224,224,3)).astype(np.float32)
      image_array = np.asarray(inp)
      normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
      data[0] = normalized_image_array

      #Getting prediction data from the array
      prediction = model.predict(data)
      gest = gesture[np.argmax(prediction)]
      print(gest)

      #Find all hands in the scene, and find the base position of the palm to determine vertical and horizontal of the hand
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        data_point = hand_landmarks.landmark[0]
        x=data_point.x
        y=data_point.y
        z=data_point.z
      print(str(x)+","+str(y)+","+str(z))
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
