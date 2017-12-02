'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 iterations
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
# imports required 
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#defining your batch size
batch = 128
# defining the number of classes 
NumberOfClasses = 10
## defining the number of iterations or iterations 
iterations = 14
# input image dimensions in this case 28 rows * 28 cols 
img_rows, img_cols = 28, 28
# the data, shuffled and split
(trainX, trainY), (testX, testY) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
    testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
    shape = (1, img_rows, img_cols)
else:
    trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
    testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
    shape = (img_rows, img_cols, 1)

# Sets trainX to a float32 type as it is most efficient for keras on gpu and cpu
trainX = trainX.astype('float32')
testX = testX.astype('float32')

# Scales the inputs(trainX,testY) to be in a range of 0-1
trainX /= 255
testX /= 255

# Print out the trainX shape which should be (60000, 28, 28, 1)
print('trainX shape:', trainX.shape)

# Print the amount of train samples used 
print(trainX.shape[0], 'train samples')
#Print the amount of test samples used
print(testX.shape[0], 'test samples')

# convert class vectors to binary class matrices. 
#Converts a class vector (integers) to binary class matrix.
#E.g. for use with categorical_crossentropy.
#Allows us to match our binary matrices to our train and test answers(0 to 9)
trainY = keras.utils.to_categorical(trainY, NumberOfClasses)
testY = keras.utils.to_categorical(testY, NumberOfClasses)

# Create a sequential Model. A sequential model is a linear stack of layers. We can then add layers to our model using the add() method.
model = Sequential()



