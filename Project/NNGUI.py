from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import glob
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
import os
# reading the NN 
import pickle
import dill
import matplotlib.pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from PIL import ImageTk, Image
import cv2










##########################################################
##                       Variables                      ##
##########################################################

# global vars
windowWidth = 1000 
minWidth = 200
windowHeight = 600
numberNeuralNetworkList = []
catdogNeuralNetworkList = []
# currently acitve NN
neuralNetwork = []
# checks if the number or the catdog NN is active
currentActiveNN = 0
# slider for backquery
onodeSliderList=[]
# filepath to query a picture
pictureFilename = ""
# the "name" of the ouputNodes from the NN
onodeNames = [["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],["cat", "dog"]]


# paths
# https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
# relative path to files
fileDir = os.path.dirname(os.path.realpath('__file__'))
training_data_path = os.path.join(fileDir, '../../trainingdata/testdata_mnist/mnist_train.csv')
test_data_path = os.path.join(fileDir, '../../trainingdata/testdata_mnist/mnist_test.csv') 
trained_Number_NN_Path = os.path.join(fileDir, 'Trained_NN/Numbers/numberNN*') 
trained_CatDog_NN_Path = os.path.join(fileDir, 'Trained_NN/CatDog/catdogNN*') 










####################################################
##                       GUI                      ##
####################################################


####################### Initialize ########################

# initialize the GUI
root = Tk()
root.title("Neural Network Visualizer")
root.geometry("1000x600")


# create the NN buttons at the top
button_chooseNN_number = Button(root, text="Neural Network Numbers", command = lambda: openNN_Number())
button_chooseNN_catdog = Button(root, text="Neural Network Cat Or Dog", command=lambda: openNN_CatDog())

# add the NN buttons to the grid
button_chooseNN_number.grid(row=0, column=0, columnspan=2)
button_chooseNN_catdog.grid(row=0, column=2, columnspan=2)


# create a label for the notebook
notebook_label = Label(root)

# place the label for the notebook
notebook_label.grid(row=1, column=0, columnspan=4)

# create the notebook for the tabs
notebook = ttk.Notebook(notebook_label)
notebook.pack()


####################### Tabs ########################

# create the tabs
frame_chooseTrainedNN = Frame(notebook, width=windowWidth, height=windowHeight, bg="")
frame_queryPicture = Frame(notebook, width=windowWidth, height=windowHeight, padx=20, pady=20)
frame_drawPicture = Frame(notebook, width=windowWidth, height=windowHeight)
frame_backqueryNN = Frame(notebook, width=windowWidth, height=windowHeight, padx=20, pady=20)

frame_chooseTrainedNN.pack(fill=BOTH, expand=TRUE)
frame_queryPicture.pack(fill=BOTH, expand=TRUE)
frame_drawPicture.pack(fill=BOTH, expand=TRUE)
frame_backqueryNN.pack(fill=BOTH, expand=TRUE)

# add the tabs to the notebook
notebook.add(frame_chooseTrainedNN, text="choose a trained NN")
notebook.add(frame_queryPicture, text="query a picture")
notebook.add(frame_drawPicture, text="draw a picture")
notebook.add(frame_backqueryNN, text="backquery the NN")


####################### frame_chooseTrainedNN Layout ########################

# add a label to frame_chooseTrainedNN for the current NN
chose_NN_Current = Label(frame_chooseTrainedNN, padx=10, pady=20, text="no NN chosen currently")
chose_NN_Current.pack(fill=X)

# add a Frame to frame_chooseTrainedNN for available NN
choose_NN_Available_Frame = Frame(frame_chooseTrainedNN, bg="green")
choose_NN_Available_Frame.pack(fill=X)
# create a scrollbar for the available NN
choose_NN_Available_scrollbar = Scrollbar(choose_NN_Available_Frame)
# create a Listbox for the available NN
choose_NN_Available_Listbox = Listbox(choose_NN_Available_Frame,height=25, yscrollcommand=choose_NN_Available_scrollbar.set)
# configure the scrollbar
choose_NN_Available_scrollbar.config(command=choose_NN_Available_Listbox.yview)
# add scrollbar 
choose_NN_Available_scrollbar.pack(side=RIGHT, fill=Y)
# add Listbox
choose_NN_Available_Listbox.pack(fill=BOTH, expand=TRUE)

# create a Button to select a NN
choose_NN_Available_Button = Button(frame_chooseTrainedNN, text="select NN", padx=20, pady=5, command=lambda: loadNN())
# add the button
choose_NN_Available_Button.pack(side=TOP)


####################### frame_queryPicture Layout ########################

# create a Button to select a file
select_Picture_Dialog_Button = Button(frame_queryPicture, text="select a picture", padx=20, pady=5, command=lambda: selectPicture())
# add the button
select_Picture_Dialog_Button.pack(side=TOP)
# add a label with the picture path
selected_Picture_Label = Label(frame_queryPicture, padx=10, pady=20, text="no picture chosen currently")
# add the label
selected_Picture_Label.pack(side=TOP, fill=X)

# create a Button to query the picture
queryPicture_Button = Button(frame_queryPicture, text="query the picture", padx=20, pady=5, command=lambda: queryPicture())
# add the button
queryPicture_Button.pack(side=TOP)

# add a Frame to frame_chooseTrainedNN for available NN
queryPicture_Result_Frame = Frame(frame_queryPicture, padx=20, pady=30)
queryPicture_Result_Frame.pack(fill=X)









##########################################################
##                       Functions                      ##
##########################################################


####################### for the NN buttons ###################################

def openNN_Number():
    global currentActiveNN, onodeNameCurrent
    currentActiveNN = 0
    return

def openNN_CatDog():
    global currentActiveNN, onodeNameCurrent
    currentActiveNN = 1
    return


####################### for frame_chooseTrainedNN tab ########################

def getAllNN():
    print("getAllNN")
    # for every numberNN file in the folder
    for trained_NN_File in glob.glob(trained_Number_NN_Path):
        print("getAllNN current file: ",trained_NN_File)
        # open the file and save the object 
        with open(trained_NN_File, 'rb') as input:
            # append it to the list of NNs
            numberNeuralNetworkList.append(dill.load(input))

    # for every catdogNN file in the folder
    for trained_NN_File in glob.glob(trained_CatDog_NN_Path):
        # open the file and save the object 
        with open(trained_NN_File, 'rb') as input:
            # append it to the list of NNs
            catdogNeuralNetworkList.append(pickle.load(input))


def displayNN(nnList):
    print("displayNN")
    # clear the old Listbox entries
    choose_NN_Available_Listbox.delete(0,END)

    # iterate through the array and display the NNs
    for nn in nnList:
        # display text on the List like (nodes: 100   learningrate: 0.2   trainingepochs: 2   performance: 0.98   )
        displayTextChooseNNButton = "nodes: %s \t  learningrate: %s   trainingepochs: %s   performance: %s   "%(nn.hnodes, nn.lr, nn.epochs, nn.performance)
        # add the text to the listbox
        choose_NN_Available_Listbox.insert(END, displayTextChooseNNButton)


def loadNN():
    global neuralNetwork
    # get the index of the select List entry
    loadIndex = choose_NN_Available_Listbox.curselection()[0]
    print("loadNN index: ", loadIndex)
    # the index is retrieved and the NN gets chosen out of the current NNList
    if(currentActiveNN==0):
        print("number is activ")
        neuralNetwork = numberNeuralNetworkList[loadIndex]
    else:
        neuralNetwork = catdogNeuralNetworkList[loadIndex]
    # reload 
    reloadForNewNN()


####################### for frame_queryPicture tab ########################


def selectPicture():
    global pictureFilename
    # select a file path over dialog
    pictureFilename = filedialog.askopenfilename(initialdir=fileDir, title="select an image to query", filetypes=(("png","*.png"),("jpg","*.jpg")))
    # print the path on the GUI
    selected_Picture_Label.configure(text="current path  ->  "+pictureFilename)
    # add image to the GUI
    selected_Picture = ImageTk.PhotoImage(Image.open(pictureFilename))
    display_Selected_Picture_Label = Label(frame_queryPicture, image=selected_Picture)
    display_Selected_Picture_Label.pack()


def queryPicture():
    # prepare the picture
    imageToQuery = preparePictureForNN(pictureFilename)
    # query the picture
    resultList = neuralNetwork.query(imageToQuery)
    # show results
    showQueryResult(resultList)


def showQueryResult(resList):
    # clear the old results
    # https://stackoverflow.com/questions/50656826/how-can-i-delete-the-content-in-a-tkinter-frame/50657381
    for widget in queryPicture_Result_Frame.winfo_children():
        widget.destroy()
    
    # the index of the highest value corresponds to the label
    result = numpy.argmax(resList)
    # add label with the result
    resultLabel = Label(queryPicture_Result_Frame, pady=20, text="neural Network thinks it's a %s"%(onodeNames[currentActiveNN][result]))
    # show label
    resultLabel.pack(fill=X)

    # add for every calculated onode a value
    for index, result in enumerate(resList):
        # add label with node and result
        onodeResultLabel = Label(queryPicture_Result_Frame, text="output node %s:   %.3f%%"%(onodeNames[currentActiveNN][index], 100*result))
        # show label
        onodeResultLabel.pack(fill=X)
    
def preparePictureForNN(picturePath):
    # define the image size (only the square is given)
    imgSize = int(math.sqrt(neuralNetwork.inodes))
    # get image from path
    originalImage = cv2.imread(picturePath)
    # resize the image with openCv2
    resizedImage = cv2.resize(originalImage, (imgSize, imgSize), interpolation=cv2.INTER_NEAREST)
    # gray the image
    resizedGrayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
    # load image data into an array
    img_array = resizedGrayImage.flatten()
    # reshape from resized to input_nodes value, invert values
    img_data  = 255.0 - img_array.reshape(neuralNetwork.inodes)
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    # return the changed image data
    return img_data

####################### for frame_drawPicture tab #########################



####################### for frame_backqueryNN tab #########################

def init_BackqueryNN_Tab():
    global onodeSliderList
    global backquery_Picture_Frame

    onodeSliderList=[]
    # iterate through the array and display the NNs
    for onode in range(neuralNetwork.onodes):
        # add label to specifiy which output node is altered
        onodeSliderLabel = Label(frame_backqueryNN, text="output node %s"%(onodeNames[currentActiveNN][onode]))
        # show label
        onodeSliderLabel.grid(row=onode, column=0)
        # create slider and add it to the array
        onodeSliderList.append(Scale(frame_backqueryNN, from_=1, to=99, length=minWidth, orient=HORIZONTAL))
        # show slider
        onodeSliderList[onode].grid(row=onode, column=1)
    
    # create the button to compute
    button_Compute_Backquery = Button(frame_backqueryNN, padx=20, pady=5, text="compute the backquery", command = lambda: compute_Backquery())
    # add the button to the grid
    button_Compute_Backquery.grid(row=neuralNetwork.onodes, column=0, columnspan=2)
    #button_Compute_Backquery.place(anchor="center")
        
    # create the Frame to hold the picture
    backquery_Picture_Frame = Frame(frame_backqueryNN, bg="black")
    # add the Frame to the grid
    backquery_Picture_Frame.grid(row=0, column=2, padx=40, rowspan=neuralNetwork.onodes)

def compute_Backquery():
    # clear the old picture
    # https://stackoverflow.com/questions/50656826/how-can-i-delete-the-content-in-a-tkinter-frame/50657381
    for widget in backquery_Picture_Frame.winfo_children():
        widget.destroy()

    # node value list
    output_Node_Value = []

    # get all values in order - save in the output value list
    for slider in onodeSliderList:
        # divide by 100 to get percentage
        output_Node_Value.append(slider.get()/100.0)
    print(output_Node_Value)

    # backquery and get the image data
    image_data = neuralNetwork.backquery(output_Node_Value)
    # define the image size (only the square is given)
    imgSize = int(math.sqrt(neuralNetwork.inodes))
    # plot image data
    matplotlib.pyplot.imshow(image_data.reshape(imgSize,imgSize), cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
    # show the image in the GUI
    # https://stackoverflow.com/questions/19612419/updating-matplotlib-imshow-from-within-a-tkinter-gui
    fig = matplotlib.pyplot.figure(figsize=(4,4))
    canvas = FigureCanvasTkAgg(fig, backquery_Picture_Frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)    

####################### for other #######################################

def reloadForNewNN():
    # reload the name to the new one
    chose_NN_Current.configure(text="current NN  ->  nodes: %s   learningrate: %s   trainingepochs: %s   performance: %s   "%(neuralNetwork.hnodes, neuralNetwork.lr, neuralNetwork.epochs, neuralNetwork.performance))
    # reload the backquerry sliders
    init_BackqueryNN_Tab()










#####################################################################
##                       Neural Network Class                      ##
#####################################################################

# neural network class definition
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each inpzt, hidden oputput layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate

        # link weight matrices, wih and who 
        # weights inside the arrays aer w_i_j, where link is from node i to node j in the next layer
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        # acitivation function is the sigmoid funciton
        #self.activation_function = lambda x: scipy.special.expit(x)
        #self.inverse_activation_function = lambda x: scipy.special.logit(x)
        
        # performance 
        self.performance = 0
        
        # epochs
        self.epochs = 0

        pass
    
    def activation_function(self, arr):
        arrX = 0
        for x in arr:
            arrY = 0
            for y in x:
                arr[arrX][arrY] = scipy.special.expit(y)
                arrY += 1
                pass
            arrX += 1
            pass
        return arr    

    def inverse_activation_function(self, arr):
        arrX = 0
        for x in arr:
            arrY = 0
            for y in x:
                arr[arrX][arrY] = scipy.special.logit(y)
                arrY += 1
                pass
            arrX += 1
            pass
        return arr
    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer        
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)      
        
        # calculate signals final output layer        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    
    # backquery the neural network
    # we'll use the same termnimology to each item, 
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs










######################################################################
##                       Init and Loop the GUI                      ##
######################################################################

# init all the stuff
getAllNN()
displayNN(numberNeuralNetworkList)

# start the GUI
root.mainloop()