import tensorflow as tf
import cv2
import numpy as np

# Define the eye aspect ratio function
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1]-eye[5])
    B = np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the haarcascade classifier for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the region of interest that contains the eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # For each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Extract the eye region
            eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Calculate the eye aspect ratio
            eye = np.array([(ex, ey), (ex+ew, ey), (ex+ew, ey+eh), (ex, ey+eh), (ex+ew//2, ey), (ex+ew//2, ey+eh)])
            ear = eye_aspect_ratio(eye)

            # Reshape the eye image to feed it to the model
            eye_image = cv2.resize(eye_roi_gray, (64, 64))
            eye_image = np.reshape(eye_image, (1, 64, 64, 1))

            # Predict the sleepiness using the model
            prediction = model.predict(eye_image)

            # If the prediction is high, the driver is sleepy
            if prediction > 0.5:
                cv2.putText(frame, 'Sleepy', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('frame', frame)

    # If the 'q' key is pressed, stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows

cv2.destroyAllWindows()
vs.stop()
