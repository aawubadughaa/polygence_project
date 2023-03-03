import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

import os
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
#import pdb
global testing_data, training_data

TRAIN_DIR = r"C:\Users\matth\OneDrive\vscode\polygence\dataset_new\train\train1"
TEST_DIR = r"C:\Users\matth\OneDrive\vscode\polygence\dataset_new\test\test1"
IMG_SIZE = 255
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic')

def label_img(img):
    word_label = img.split(' ')
    # DIY One hot encoder
    if word_label == '1': return [1, 0]
    elif word_label == '2': return [0, 1]

def create_train_data():
    # Creating an empty list where we should store the training data
    # after a little preprocessing of the data
    training_data = []
    # tqdm is only used for interactive loading
    # loading the training data
    for img in tqdm(os.listdir(TRAIN_DIR)):
 
        # labeling the images
        label = label_img(img)
 
        path = os.path.join(TRAIN_DIR, img)
 
        # loading the image from the path and then converting them into
        # grayscale for easier covnet prob
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
 
        # resizing the image for processing them in the covnet
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
 
        # final step-forming the training data list with numpy array of the images
        training_data.append([np.array(img), np.array(label)])
 
    # shuffling of the training data to preserve the random state of our data
    shuffle(training_data)
    #print(training_data)
    # saving our trained data for further uses if required
    np.save('train_data.npy', training_data)

def create_test_data():
    # Creating an empty list where we should store the training data
    # after a little preprocessing of the data
    testing_data = []
    # tqdm is only used for interactive loading
    # loading the training data
    for img in tqdm(os.listdir(TEST_DIR)):
 
        # labeling the images
        label = label_img(img)
 
        path = os.path.join(TEST_DIR, img)
 
        # loading the image from the path and then converting them into
        # grayscale for easier covnet prob
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
 
        # resizing the image for processing them in the covnet
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
 
        # final step-forming the training data list with numpy array of the images
        testing_data.append([np.array(img), np.array(label)])
 
    # shuffling of the training data to preserve the random state of our data
    shuffle(testing_data)
    #print(training_data)
    # saving our trained data for further uses if required
    np.save('test_data.npy', testing_data)

'''Processing the given test data'''
# Almost same as processing the training data but
# we dont have to label it.
 
'''Running the training and the testing in the dataset for our model'''
#allow_pickle = False
train_data = create_train_data()
test_data = create_test_data()

"""train_images = []
test_images = []
for i in training_data:
    x = training_data[i]/255
    train_images.append(x)

for i in testing_data:
    y = testing_data[i]/255
    test_images.append(y)
"""
class_names = ['1', '2']

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))



model.summary()



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# An epoch means training the neural network with all the
# training data for one cycle. Here I use 10 epochs
#history = model.fit(train_data, epochs=1, 
#                    validation_data=(test_data))



#plt.plot(history.history['accuracy'],label='accuracy')
#plt.plot(history.history['val_accuracy'],label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_data,
                                     
                                     verbose=2)



print('Test Accuracy is',test_acc)

