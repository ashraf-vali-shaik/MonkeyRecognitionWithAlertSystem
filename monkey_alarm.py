import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import pygameq

# Load the pre-trained MobileNetV2 model and a custom trained model for monkey detection
model = MobileNetV2(weights="imagenet")

# Initialize Pygame for playing sounds
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")  # Ensure you have an 'alarm.wav' file in your directory

# Initialize webcam
cap = cv2.VideoCapture(0)


def detect_monkey(frame):
    # Convert the frame to a blob and preprocess it
    image = cv2.resize(frame, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Predict using the pre-trained model
    preds = model.predict(image)
    P = tf.keras.applications.mobilenet_v2.decode_predictions(preds)

    # Check if 'monkey' is in the top predictions
    for (_, label, prob) in P[0]:
        if 'monkey' in label.lower() and prob > 0.5:
            return True

    return False


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if detect_monkey(frame):
        # If monkey is detected, play the alarm sound
        alarm_sound.play()

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
