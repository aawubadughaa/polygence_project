import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# define the eye aspect ratio function
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# define the function to extract eye features from a photo
def extract_eye_features(photo_file):
    img = cv2.imread(photo_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    eye_features = []
    for (x,y,w,h) in eyes:
        eye = [(x, y), (x + w//2, y - h//4), (x + w, y), (x + w//2, y + h//4), (x, y + h), (x + w//2, y + 3*h//4)]
        ear = eye_aspect_ratio(eye)
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

# reshape the data
train_eye_features = train_eye_features.reshape(train_eye_features.shape[0], train_eye_features.shape[1], 1)
test_eye_features = test_eye_features.reshape(test_eye_features.shape[0], test_eye_features.shape[1], 1)

# define the tensorflow model
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(train_eye_features.shape[1], 1, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_eye_features, train_labels, epochs=10, validation_data=(test_eye_features, test_labels))

# evaluate the model
test_loss, test_acc = model.evaluate(test_eye_features, test_labels)
print('Test accuracy:', test_acc)
