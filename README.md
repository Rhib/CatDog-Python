# CatDog Python


## Module Requirements

- Python version: 3.8.5 
- Standard libraries:
	- tkinter
	- glob
	- os
	- pickle
	- math
	- logging
	- pathlib
	- datetime
- External libraries
	- PIL
	- imageio
	- numpy
	- scipy
	- dill
	- matplotlib
	- cv2
- Tested OS
	- Windows 10
	- Ubuntu Linux

**To install the requirements type into the command prompt ```pip install -r requirements.txt```** 



## How to use the CatDog neural network:

 1. **Clone this repository**
 2. **Open a command prompt**
 3. **Navigate to the local repository**
 4. **Type: ```python Main.py``` to start the programm**



# Programm description

## Goal
The original goal of this project was to implement a neural network to decide if a picture shows a cat or a dog.
The user should be able to upload a picture into an already trained neural network and have the results displayed in a GUI
We therefore oriented ourselves at Tariq Rashids book "Neuronale Netzwerke selbst implementieren".
Tariq Rashids book gives a good introduction to neural networks and is a good refresher on matrix calculus, every calculation has to be programmed by the user itself and no frameworks like tensorflow are used.
Therefore we were unfortunately unable to handle cat and dog images with Tariq Rashids approach, because of the increased complexity.
As a result we decided to make a neural network to recognice hand written numbers and put the cat and dog neural network in the back of the line.
We implemented a graphical user interface with different nets. Thoses nets do have different performance levels. You can either upload images of handwritten numbers or make them yourself in the GUI. The neural network used test and training data to learn. It will be able to decide which number is shown, even if the conditions are bad.

## User manual
### Choose number or catdog neural networks
At the top of the window to buttons are displayed and the user can chosse which neural networks should be loaded
either the number neural networks are loaded or the cat or dog neural networks are loaded (currently we have no trained cat or dog neural networks so the numbers are loaded as default)

### Choose a trained neural network
The first option is to select an already trained neural network. The different neural networks differ in their amount of 
nodes, their learning rate, their trainingepochs and their performance. Afterwards the neural network is used for the following three options.

#### Query a picture
This option allows you to upload a picture from your locale storage. Your then have the option to query the picture.
The neural network will give it's proposal in what number the uploaded picture shows. You will get an output with the percentages with the probability of each number from 0 to 9.

#### Draw a picture
Alternatively you can draw a picuture directly in the neural network visualizer. If you messed up you have the possibility to clear the canvas. If you are pleased with your result press "query the canvas". You will also get the probabilities on which number it is.

#### Backquery the nn
Lastly you have the option to tell the neural network what the output should be and the neural network will compute from output back to input and will show you what it thinks the input is, or what it would see as the "perfect" input


## Further information

### Log files
Besides the folder of the cloned repository the programm will create an Folder "CatDog-Python-Logs". In this folder the .log files can be found

### Folder Example_Images_To_Query
In this folder some example pictuers are given which can be used in the previous described query a pictuer function

### Notebooks (not part of the submission)
The notebooks were needed to create the neural networks in the first place, but they are not part of the submission

#### NumberNN
The neural network for number recognition from Tariq Rashids book

#### NumberNN_save_trained
The script that trains the neural networks and saves them to the folder

#### CatDogNN
Our approach to recognize cats and dogs with Tariq Rashids method

