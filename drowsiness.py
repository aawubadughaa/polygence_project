import numpy as np
import cv2
import os
import dlib
from scipy.spatial import distance as dist
import tensorflow as tf

# define the eye aspect ratio function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# define the function to extract eye features from a photo
def extract_eye_features(photo_file):
    img = cv2.imread(photo_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = detector(gray, 0)
    eye_features = []

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []

        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))

        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        eye_features.append(ear)

    return eye_features

# define the function to load the data
def load_data(data_dir):
    eye_features = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            photo_file = os.path.join(data_dir, filename)
            features = extract_eye_features(photo_file)
            eye_features.append(features)

            if 'awake' in filename.lower() or 'open' in filename.lower():
                label = 0
            else:
                label = 1
            labels.append(label)

    eye_features = np.array(eye_features)
    labels = np.array(labels)
    return eye_features, labels

# load the training and testing data
train_data_dir = 'train_photos'
test_data_dir = 'test_photos'
train_eye_features, train_labels = load_data(train_data_dir)
test_eye_features, test_labels = load_data(test_data_dir)

# define the tensorflow model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_eye_features, train_labels, epochs=10, validation_data=(test_eye_features, test_labels))

# evaluate the model
test_loss, test_acc = model.evaluate(test_eye_features, test_labels)
print('Test accuracy:', test_acc)

#http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
