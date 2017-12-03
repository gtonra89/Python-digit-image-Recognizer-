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
	
@app.route('/about/')
def about():
    return render_template('about.html')

# Analayse the input given function
# we use get and post actions
@app.route('/analyse/', methods=['GET','POST'])
def analyse():
# calls the parse img function to 
# get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    parseRead = imread('output.png', mode='L')
	#Computes the bit-wise NOT of the underlying binary representation of the integers in the input arrays
    parseRead = np.invert(parseRead)
    parseRead = imresize(parseRead,(28,28))

    # reshape image data for use in neural network
    parseRead = parseRead.reshape(1,28,28,1)

    # load model to predict number
	#You can then use keras.models.load_model(filepath) to reinstantiate your model.
	#load_model will also take care of compiling the model using the saved training configuration 
	#(unless the model was never compiled in the first place).
    trainedModel = keras.models.load_model("model/mnistModel.h5")

    # Use predict function and pass image parseRead through it to get answer
    Result = trainedModel.predict(parseRead)
    print(Result)
	
	# change Result to number string
	#Return a string representation of the data in the array.
    ResString = np.array_str(np.argmax(Result, axis=1))
    print(ResString)
    return ResString
	
# Parsing Image function
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))
		
if __name__ == '__main__':
    app.run(debug = True)