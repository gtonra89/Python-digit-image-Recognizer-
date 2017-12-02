# Code adapted from https://github.com/sleepokay/mnist-flask-app

## imports required to make the system run 
from flask import Flask, render_template, request
from scipy.misc import imsave , imread, imresize
import numpy as np
import keras.models
import re
import base64

# creates an instance of the flask app
app = Flask(__name__)

# returns the renders_template of index.html to display page
@app.route('/')
def index():
    return render_template("index.html")

# Analayse the input given function
# we use get and post actions
@app.route('/analyse/', methods=['GET','POST'])
def analyse():
# get data from drawing canvas and save as image
    parseImg(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    parseRead = imread('output.png', mode='L') 
    parseRead = np.invert(parseRead)
    parseRead = imresize(parseRead,(28,28))

    # reshape image data for use in neural network
    parseRead = parseRead.reshape(1,28,28,1)

    # load model to predict number
    trainedModel = keras.models.load_model("model/mnistModel.h5")

    # Use predict function and pass image parseRead through it to get answer
    Result = trainedModel.predict(parseRead)
    print(Result)