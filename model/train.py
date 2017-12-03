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

#Now let’s talk about batch_size. It relates to how we train the model, 
#specifically how to optimize the loss function. 
#In the naive form, we compute the loss function over the whole dataset
batch = 128
# defining the number of classes 
NumberOfClasses = 10
## defining the number of iterations also epoces i rather iterations 
iterations = 14
# input image dimensions in this case 28 rows * 28 cols 
img_rows, img_cols = 28, 28
# the data, shuffled and split
(trainX, trainY), (testX, testY) = mnist.load_data()


#Since we are using CNN, one important step is to arrange the neurons in 3D (width, height and depth). 
#I’ll skip the details but the depth here in code is 1. 
#That means our images have only 1 channel, instead of 3 (RGB channels)
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
#Normally we would want to preprocess the dataset so that each feature has zero mean and unit standard deviation,
#but in this case the features are already in a nice range from -1 to 1
trainX /= 255
testX /= 255

# Print out the trainX shape
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

# Create a sequential Model. 
#A sequential model is a linear stack of layers. 
#We can then add layers to our model using the add() method.
model = Sequential()

# We add Conv2D layer. Allows spatial convolution over images. 
#This is a 2d convolution layer.
# It is the first layer in the model and it create a convolution kernal.
# All parameters are added to the layer

# 32 represents the amount of filters for your conv2d layer.

# Kernal_size receives a list/tuple of 2 integers specifying width and height of a 2D convolution window. 
# relu is a activations function. Activation functions allows us to get better accuracy. 
#Depending on which activation function you use, it can provide lower or higher gradients. Relu allows the network the max amount of the error in back propagation

# Input shape represents the elements an array has in each dimension(ie. image_rows,image_cols,1)                
model.add(Conv2D(32, 
                kernel_size=(3, 3),activation='relu'.
                shape=shape))

# Add another convolution layer with parameters (filters,kernal_size and activation function)
model.add(Conv2D(64, (3, 3), activation='relu'))

# The MaxPooling2D layer is used for downsampling the image size.
#a (2,2) pool it splits a pixel image into 4 chucks
# and takes the 4 highest values from each chunk to represent or (2,2) pool
model.add(MaxPooling2D(pool_size=(2, 2)))

# A Dropout method consists in randomly setting a fraction rate of input units to 0 
# at each update during training time, which helps prevent overfitting.
model.add(Dropout(0.25))

# it flattens the input provided
model.add(Flatten())

# Add another a dense layer with a output of 128 (nodes) & relu yet again
model.add(Dense(128, activation='relu'))

# A Dropout method consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
model.add(Dropout(0.5))

# Apply a final dense layer with a output of num_class(10) which is our number of outcomes and use activation function softmax. We apply the softmax activation function to the final layer as it can be used to represent catagorical data. Outputing results ranging from 0 upto 1
model.add(Dense(NumberOfClasses, activation='softmax'))

# Compile method configures our model before training. Its takes arguments: Optimizer, loss and metrics
model.compile(loss=keras.losses.categorical_crossentropy,# Loss(objective) Function. Categorical_crossentrophy: catagorises our loss with our catagories
              optimizer=keras.optimizers.Adadelta(),# Adadelta Optimizer used to leave perametersat their default values. Better algorithm then the classic gradient descent
              metrics=['accuracy'])# Used to judge the performance of your model
# To train our model we use the fit function

# To train our model we use the fit function
model.fit(trainX, trainY,
          batch=batch,# Batch_size will give us samples per gradient update
          iterations=iterations, # the number of iterations it runs
          verbose=1, #a progress bar of the training when set to 1
          validation_data=(testX, testY)) # Evaluate the loss and model metrics of each iteration
# Calculate our loss and accuracy of our test data.
score = model.evaluate(testX, testY, verbose=0)# Input and output and verbose set to 0 means silent mode

# Print out Loss and accuracy of our test set.
print('The Test loss is:', score[0])
print('The Test accuracy is:', score[1])

# Save Model in order to be reused again.
model.save("mnistModel.h5")



