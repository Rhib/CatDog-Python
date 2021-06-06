# CatDog Python

## Module Requirements

### Tested OS

- Windows 10
- Ubuntu Linux

### Module requirements

- From python 3.8.5 standard library:
	- tkinter
	- glob
	- numpy
	- scipy
	- os
	- pickle
	- dill
	- matplotlib
	- math
	- cv2
	- logging
	- datetime

**To install the requirements type into the command prompt ```pip install -r requirements.txt```** 

## How to use the CatDog neural network:

 1. **Clone this repository**
 2. **Open a command prompt**
 3. **Navigate to the local repository**
 4. **To execute the python script type in: ```python main.py```**

## Game description
### Goal
Our goal was to implement a neural network to decide if a picture shows a cat or a dog.
We therefore oriented ourselves at Tariq Rashids book "Neuronale Netzwerke selbst implementieren".
Unfortunately, cat and dog images were way to komplex to be handled with our neuronal network.
Therefore we decided to make a neural network to recognice hand written numbers. We implemented a graphical user interface with different nets. Thoses nets do have different performance levels. You can either upload images of handwritten numbers or make them yourself in the GUI. The neural network used test and training data to learn. It will be able to decide which number is shown, even if the conditions are bad.

# User manual
## Usage
## Choose a trained neural network
The first option is to select an already trained neural network. The defferent neural networks differ in their amount of 
nodes, their learning rate, their trainingepochs and their performance. Afterwards the neural network is used for the following three options.

## Query a picture
This option allows you to upload a picture from your locale storage. Your then have the option to query the picture.
The neural network will then give it's proposal in what number the uploaded picture shows. You will get an output with the percentages with the probability of each number from 0 to 9.

## Draw a picture
Alternatively you can draw a picuture directly in the neural network visualizer. If you messed up you have the possibility to clear the canvas. If you are finished just press "query the canvas". You will also get the probabilities on which number it is.\


## Backquery the nn
Depending on the received trainingsdata you have the opportunity to let the neural network show what the "perfekt" number looks like. For example you can let you show a perfekt zero depending on the training data.

