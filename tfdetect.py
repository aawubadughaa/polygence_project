import tensorflow as tf
import cv2
import numpy as np

def eye_aspect_ratio(eye):
    if len(eye) == 0:
        return 0.0
    
    A = np.linalg.norm(eye[1]-eye[5])
    B = np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])
    
    ear = (A + B) / (2.0 * C)
    
    return ear




# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the haarcascade classifier for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')\

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame to the required input size of the model
    resized_frame = cv2.resize(frame, (64, 64))

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    # For each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the region of interest that contains the eyes
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the region of interest
        eyes = eye_cascade.detectMultiScale(roi_color)

        # Initialize variables for drowsiness detection
        ear_thresh = 1.35  # EAR threshold for drowsiness
        consec_frames = 30  # Number of consecutive frames below threshold to trigger alarm
        ear_frame_count = 0  # Count of frames below threshold
        alarm_on = False  # Flag to indicate if alarm is on

        # For each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

            # Get the eye image and preprocess it
            eye_image = roi_color[ey:ey+eh, ex:ex+ew]
            eye_image = cv2.resize(eye_image, (64, 64))
            eye_image = eye_image.astype("float") / 255.0  # convert to float

            # Make a prediction using the model
            prediction = model.predict(np.expand_dims(eye_image, axis=0), verbose=0)[0][0]

            # Calculate the eye aspect ratio and print the prediction
            ear = eye_aspect_ratio(roi_color[ey:ey+eh, ex:ex+ew])
            #print(prediction)
            print(ear)
            # Check if EAR is below threshold
            if ear < ear_thresh:
                ear_frame_count += 1
                # Check if consecutive frames below threshold
                if ear_frame_count >= consec_frames:
                    # Trigger alarm or take other action
                    if not alarm_on:
                        # turn on alarm or take other action
                        print("Drowsiness detected!")
                        alarm_on = True
            else:
                ear_frame_count = 0
                if alarm_on:
                    # turn off alarm or reset action
                    alarm_on = False

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
