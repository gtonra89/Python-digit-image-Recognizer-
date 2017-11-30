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

# returns the renderstemplate of index.html
@app.route('/')
def index():
    return render_template("index.html")

# Predict function
@app.route('/analyse/', methods=['GET','POST'])
def analyse():

   