import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the eye aspect ratio (EAR) threshold to detect drowsiness
EAR_THRESHOLD = 0.2

# Define a function to calculate the EAR of each eye
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1]-eye[5])
    B = np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])
    return (A+B)/(2.0*C)

# Initialize the video stream and grab the first frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Define the initial values for the EAR and consecutive frames with closed eyes
EAR = 0
COUNTER = 0

# Start the main loop
while True:
    # Read a new frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Loop over the faces
    for face in faces:
        # Detect the landmarks of the face
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # Extract the left and right eye landmarks and calculate the EAR for each eye
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)

        # Calculate the average EAR for both eyes
        EAR = (left_EAR + right_EAR) / 2.0

        # Check if the EAR is below the threshold to detect drowsiness
        if EAR < EAR_THRESHOLD:
            # Increment the counter for consecutive frames with closed eyes
            COUNTER += 1

            # If the counter exceeds a certain threshold, alert the user that they are drowsy
            if COUNTER >= 25:
                cv2.putText(frame, "DROWSY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Reset the counter if the eyes are open
            COUNTER = 0

        # Draw the landmarks and the EAR for both eyes on the frame
        cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)
        cv2.putText(frame, "EAR: {:.2f}".format(EAR), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Check for the "q" key to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()