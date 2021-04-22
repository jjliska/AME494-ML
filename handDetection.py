import cv2
import mediapipe as mp
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from pysinewave import SineWave
import collections

class pixels:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

gesture = ['Paper','Rock','Scissors']
gest = ""

pixelLoc = []

camRes = [1280,720]

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, camRes[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camRes[1])
# Loading the model trained from Teachable Machine
model = tensorflow.keras.models.load_model('keras_model.h5')

#Create sinewave
sinewave = SineWave(pitch = 0, pitch_per_second = 100)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    if gest == 'Paper':
        color = 'red'
        maxPitch = 24
    elif gest == 'Rock':
        color = 'green'
        maxPitch = 12
    else:
        color = 'blue'
        maxPitch = 6
        pass
    #Start Sine Wave
    sinewave.play()

    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    # Getting image 224x224 image before we flip color and screen
    fc = cv2.resize(image, (224,224))
    # Inverting color of screen
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      # Creating array
      data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
      inp = np.reshape(fc,(1,224,224,3)).astype(np.float32)
      # Setting image to array
      image_array = np.asarray(inp)
      normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
      data[0] = normalized_image_array
      # Plugging numericals into the prediction model
      prediction = model.predict(data)
      # Getting the String attatched to gesture
      gest = gesture[np.argmax(prediction)]

      #Getting the first hand rendered in the screen
      hand_landmark = results.multi_hand_landmarks[0]
      data_point = hand_landmark.landmark[0]

      # Getting positional data of the wrist
      # Coordinate plane between 0-1 x 0-1
      x=data_point.x
      y=data_point.y
      pixelLoc.append(pixels(x,y,color))
      for i in range(len(pixelLoc)):
        cv2.line(image, (int(pixelLoc[i-1].x*camRes[0]),int(pixelLoc[i-1].y*camRes[1])), (int(pixelLoc[i].x*camRes[0]),int(pixelLoc[i].y*camRes[1])), (0,255,0), 2)
      print(gest+" @ ("+str(x)+","+str(y))
      #Setting pitch and volume given screen position of wrist
      sinewave.set_pitch(maxPitch*(x-.5))
      sinewave.set_volume(8*(y-.5))
    # Showing frame
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
