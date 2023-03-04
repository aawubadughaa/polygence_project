import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
sess = tf.compat.v1.Session(config=config)
import cv2
import numpy as np

# Define the eye aspect ratio function
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1]-eye[5])
    B = np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the haarcascade classifier for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the images of open eyes
open_eye_dir = '/Users/matthew/Library/CloudStorage/OneDrive-Personal/vscode/Polygence Project/train 2/Open_Eyes'
open_eye_images = []
for filename in os.listdir(open_eye_dir):
    img = cv2.imread(os.path.join(open_eye_dir, filename))
    if img is not None:
        img_resized = cv2.resize(img, (64, 64))
        open_eye_images.append(img_resized)

# Load the images of closed eyes
closed_eye_dir = '/Users/matthew/Library/CloudStorage/OneDrive-Personal/vscode/Polygence Project/train 2/Closed_Eyes'
closed_eye_images = []
for filename in os.listdir(closed_eye_dir):
    img = cv2.imread(os.path.join(closed_eye_dir, filename))
    if img is not None:
        img_resized = cv2.resize(img, (64, 64))
        closed_eye_images.append(img_resized)

# Create the labels for the images
open_eye_labels = np.zeros(len(open_eye_images))
closed_eye_labels = np.ones(len(closed_eye_images))
labels = np.concatenate([open_eye_labels, closed_eye_labels])

# Combine the images into a single dataset
images = np.concatenate([open_eye_images, closed_eye_images])
images = images / 255.0

# Shuffle the dataset
p = np.random.permutation(len(images))
images = images[p]
labels = labels[p]

# Split the dataset into training and validation sets
split = int(len(images) * 0.8)
train_images = images[:split]
train_labels = labels[:split]
val_images = images[split:]
val_labels = labels[split:]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Save the model
model.save('model.h5')
