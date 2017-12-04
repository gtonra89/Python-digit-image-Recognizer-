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


# Specifically how to optimize the loss function. 
batch = 128

NumberOfClasses = 10 # there are 10 classes (1 per digit)

# defining the number of iterations 
iterations = 14

img_rows, img_cols = 28, 28 # MNIST images are 28x28 and greyscale

# the data, shuffled and split
(trainX, trainY), (testX, testY) = mnist.load_data()


#Since we are using CNN, one important step is to arrange the neurons in 2D (width, height)  
#That means our images have only 1 channel, instead of 3 (RGB channels)
if K.image_data_format() == 'channels_first':
    trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
    testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
    shape = (1, img_rows, img_cols)
else:
    trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
    testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
    shape = (img_rows, img_cols, 1)

# Sets trainX to a float32 type as it is most efficient for keras
trainX = trainX.astype('float32') 
testX = testX.astype('float32')

trainX /= 255 # Normalise data to [0, 1] range
testX /= 255 # Normalise data to [0, 1] range

print('trainX shape:', trainX.shape)

# Print the amount of samples 
print(trainX.shape[0], 'train samples')
print(testX.shape[0], 'test samples')

 
#Converts a class vector (integers) to binary class matrix.
#Allows us to match our binary matrices to our train and test answers(0 to 9)
trainY = keras.utils.to_categorical(trainY, NumberOfClasses)
testY = keras.utils.to_categorical(testY, NumberOfClasses)


#A sequential model is a linear stack of layers.
model = Sequential()


#This is a 2d convolution layer. It is the first layer in the model and it creates a convolution kernal.
# 32 amount of filters for conv2d layer.
#list 2 ints width and height, relu is a activations function
# Input shape represents the elements an array has in each dimension              
model.add(Conv2D(32, 									
                kernel_size=(3, 3),activation='relu'. 	
                shape=shape))

# Add another convolution layer with parameters (filters,kernal_size and activation function)
model.add(Conv2D(64, (3, 3), activation='relu'))

# The MaxPooling2D layer is used for downsampling (2,2) pool it splits a pixel image into 4 chucks 
#takes the 4 highest values from each chunk to represent
model.add(MaxPooling2D(pool_size=(2, 2)))

# A Dropout method consists in randomly setting a fraction rate of input units to 0 which helps prevent overfitting.
model.add(Dropout(0.25))

# it flattens the input provided
model.add(Flatten())

# Add another a dense layer with a output of 128 (nodes) & relu yet again
model.add(Dense(128, activation='relu'))

# A Dropout method consists in randomly setting a fraction rate of input units to 0 at each update during training time
model.add(Dropout(0.5))

# Apply a final dense layer with a output of num_class(10), Output softmax layer ranging from 0 to 1
model.add(Dense(NumberOfClasses, activation='softmax')) 

# Compile method configures our model before training
model.compile(loss=keras.losses.categorical_crossentropy,	# reporting the accuracy
              optimizer=keras.optimizers.Adadelta(),		# Better algorithm than the classic gradient descent
              metrics=['accuracy'])							# the performance of your model

# To train our model we use the fit function
model.fit(trainX, trainY,
          batch=batch,			 							# Batch will give us samples per gradient update
          iterations=iterations, 							# the number of iterations it runs
          verbose=1, 										# a progress bar of the training when set to 1
          validation_data=(testX, testY)) 					# Evaluate the loss and model metrics of each iteration

# Calculate our loss and accuracy of our test data.
score = model.evaluate(testX, testY, verbose=0)				# verbose set to 0 means silent mode

# Print out Loss and accuracy of our test set.
print('The Test loss is:', score[0])
print('The Test accuracy is:', score[1])

# Save Model in order to be reused again.
model.save("mnistModel.h5")



