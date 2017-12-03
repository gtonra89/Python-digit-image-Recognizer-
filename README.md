# Python-digit-image-Recognizer-
4th Year Emerging Technologies Flask/Python Web App to recognize Digits From Images
The following are instructions to complete the project for the module Emerging Technologies for 2017.
## Overview

In this project you will create a web application in Python to recognise digits in images. Users will be able to visit the web application through their browser, submit (or draw) an image containing a single digit, and the web application will respond with the digit contained in the image. You should use tensorflow and flask to do this. Note that accuracy of approximately 99% is considered excellent in recognising digits, so it is okay if your algorithm gets it wrong sometimes.

## Instructions

Create a git repository with a README.md and an appropriate gitignore file. The README should explain who you are, why you created the application, how you created it, how to download and run it, and summarise any references you have used.
In the repository,<br>create a web application that serves a HTML page as the root resource. The page should contain an input where the user can upload (or draw) an image containing a digit, and an area to display the image and the digit.
Add a route to your application that accepts requests containing a user input image and responds with the digit.
Connect the HTML page to the route using AJAX.

## About me
My name is <a href ="https://github.com/gtonra89">Garret Tonra</a> I am a 4th year software development Studying in the Galway-Mayo Intitute of Technology<br>
This is one of 5 modules i am Working on this semester it focuses mainly on emerging technologies and how these technologies can be used for industry based usage
## Why I created this application
The main reason Why i chose this application is to produce a result for the module covered.<br>To produce a web app that will show the skills obtained in this module and use them to produce a useful Applictaion.<br>As well as that it will hopefully help with my Final Year Project that I am working on with two other students in my year.We are trying to use machine learning to predict trends so this module is very helpful   

## Prerequisites
Python 3.6 or  above : https://www.python.org/downloads/
## How to Run
Download the zip directory to your local computer and extract the files<br>
Open your Command Prompt of your Operating System<br>
Cd or change directory to where the folder is extracted too<br>
Then type the following command  
```python run.py```and hit Enter<br>
navigate to localhost on your web browser to see the webapp in use<br>
<a href = "127.0.0.1:5000">127.0.0.1:5000</a>

## Technolodies Used
<a href ="http://flask.pocoo.org/"><img src="https://github.com/gtonra89/Python-digit-image-Recognizer-/blob/master/pocoo_flask-card.png?raw=true" align="left"></a><br><br><br><br><br>
Flask is a micro web framework written in Python and based on the Werkzeug toolkit and Jinja2 template engine. It is BSD licensed.<br>

Flask is called a micro framework because it does not require particular tools or libraries.It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions.<br>However, Flask supports extensions that can add application features as if they were implemented in Flask itself.<br>Extensions exist for object-relational mappers, form validation, upload handling, various open authentication technologies and several common framework related tools.<br> Extensions are updated far more regularly than the core Flask program

<a href ="https://en.wikipedia.org/wiki/Ajax_(programming)"><img src = "https://github.com/gtonra89/Python-digit-image-Recognizer-/blob/master/AJAX_logo_by_gengns.svg_.png?raw=true" align="left"></a><br><br><br><br><br>
Ajax is a set of Web development techniques using many Web technologies on the client side to create asynchronous Web applications. With Ajax, Web applications can send data to and retrieve from a server asynchronously (in the background) without interfering with the display and behavior of the existing page. By decoupling the data interchange layer from the presentation layer, Ajax allows for Web pages, and by extension Web applications, to change content dynamically without the need to reload the entire page. In practice, modern implementations commonly substitute JSON for XML due to the advantages of being native to JavaScript.

<a href ="https://www.tensorflow.org/"><img src = "https://github.com/gtonra89/Python-digit-image-Recognizer-/blob/master/logo.png?raw=true" align="left"></a><br><br><br><br><br>
TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and also used for machine learning applications such as neural networks.It is used for both research and production at Google, often replacing its closed-source predecessor, DistBelief.

TensorFlow was developed by the Google Brain team for internal Google use. It was released under the Apache 2.0 open source license on November 9, 2015

<a href ="https://keras.io/"><img src = "https://github.com/gtonra89/Python-digit-image-Recognizer-/blob/master/keras-logo-2018-large-1200.png?raw=true" align="middle"></a><br><br>
Keras is an open source neural network library written in Python. It is capable of running on top of MXNet, Deeplearning4j, Tensorflow, CNTK or Theano. Designed to enable fast experimentation with deep neural networks, it focuses on being minimal, modular and extensible. It was developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System), and its primary author and maintainer is François Chollet, a Google engineer.

In 2017, Google's TensorFlow team decided to support Keras in TensorFlow's core library. Chollet explained that Keras was conceived to be an interface rather than an end-to-end machine-learning framework. It presents a higher-level, more intuitive set of abstractions that make it easy to configure neural networks regardless of the backend scientific computing library.Microsoft has been working to add a CNTK backend to Keras as well and the functionality is currently in beta release with CNTK v2.0

## Neural Networks
Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text or time series, must be translated. In this project we give a neural network the MNIST dataset as input in order to for it to perform deep learning on the images.

## References 
Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py 

Adapted the main ocomponets from this : https://github.com/sleepokay/mnist-flask-app

