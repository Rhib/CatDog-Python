from tkinter import *
from tkinter import ttk
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

# global vars
windowWidth = 1000 
minWidth = 200
windowHeight = 600
numberNeuralNetworkList = []
catdogNeuralNetworkList = []
# currently acitve NN
neuralNetwork = []
# checks if the number or the catdog NN is active
numberAcitve = TRUE
# slider for backquery
onodeSliderList=[]


# paths
# https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
# relative path to files
fileDir = os.path.dirname(os.path.realpath('__file__'))
training_data_path = os.path.join(fileDir, '../../trainingdata/testdata_mnist/mnist_train.csv')
test_data_path = os.path.join(fileDir, '../../trainingdata/testdata_mnist/mnist_test.csv') 
trained_Number_NN_Path = os.path.join(fileDir, 'Trained_NN/Numbers/numberNN*') 
trained_CatDog_NN_Path = os.path.join(fileDir, 'Trained_NN/CatDog/catdogNN*') 

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

# create the tabs
frame_chooseTrainedNN = Frame(notebook, width=windowWidth, height=windowHeight, bg="")
frame_queryPicture = Frame(notebook, width=windowWidth, height=windowHeight)
frame_drawPicture = Frame(notebook, width=windowWidth, height=windowHeight)
frame_backqueryNN = Frame(notebook, width=windowWidth, height=windowHeight)

frame_chooseTrainedNN.pack(fill="both", expand=1)
frame_queryPicture.pack(fill="both", expand=1)
frame_drawPicture.pack(fill="both", expand=1)
frame_backqueryNN.pack(fill="both", expand=1)

# add the tabs to the notebook
notebook.add(frame_chooseTrainedNN, text="choose a trained NN")
notebook.add(frame_queryPicture, text="query a picture")
notebook.add(frame_drawPicture, text="draw a picture")
notebook.add(frame_backqueryNN, text="backquery the NN")


# add a label to frame_chooseTrainedNN for the current NN
chose_NN_Current = Label(frame_chooseTrainedNN, padx=10, pady=20, text="no NN chosen currently")
chose_NN_Current.pack()
# add a Frame to frame_chooseTrainedNN for available NN
chose_NN_Avialable_Frame = Frame(frame_chooseTrainedNN)
chose_NN_Avialable_Frame.pack()





####################### Functions ########################


####################### for the NN buttons ###################################

def openNN_Number():
    global numberAcitve
    numberAcitve = TRUE
    return

def openNN_CatDog():
    global numberAcitve
    numberAcitve = FALSE
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
    # clear the old ones
    # https://stackoverflow.com/questions/50656826/how-can-i-delete-the-content-in-a-tkinter-frame/50657381
    for widget in chose_NN_Avialable_Frame.winfo_children():
        widget.destroy()

    # keep track of the index to know which to load later 
    index = 0
    # iterate through the array and display the NNs
    for nn in nnList:
        # display text on the button like (nodes: 100   learningrate: 0.2   trainingepochs: 2   performance: 0.98   )
        displayTextChooseNNButton = "nodes: %s   learningrate: %s   trainingepochs: %s   performance: %s   "%(nn.hnodes, nn.lr, nn.epochs, nn.performance)
        print("displayNN add Button: ",displayTextChooseNNButton)
        # create the button - in the lambda function the index gets saved 
        choose_NN_Button_List = Button(chose_NN_Avialable_Frame, text=displayTextChooseNNButton, padx=20, pady=3, command=lambda giveIndex=index: loadNN(giveIndex))
        # add the button
        choose_NN_Button_List.pack()
        index += 1
    

def loadNN(loadIndex):
    global neuralNetwork
    print("loadNN index: ", loadIndex)
    # the index is retrieved and the NN gets chosen out of the current NNList
    if(numberAcitve==TRUE):
        print("number is activ")
        neuralNetwork = numberNeuralNetworkList[loadIndex]
    else:
        neuralNetwork = catdogNeuralNetworkList[loadIndex]
    # reload 
    reloadForNewNN()


####################### for frame_queryPicture tab ########################



####################### for frame_drawPicture tab #########################



####################### for frame_backqueryNN tab #########################

def init_BackqueryNN_Tab():
    global onodeSliderList
    global backquery_Picture_Frame
    # keep track of the index to know which to load later 
    onodeSliderList=[]
    # iterate through the array and display the NNs
    for onode in range(neuralNetwork.onodes):
        # add label to specifiy which output node is altered
        onodeSliderLabel = Label(frame_backqueryNN, text="output node nr.%s"%(onode))
        # show label
        onodeSliderLabel.grid(row=onode, column=0)
        # create slider and add it to the array
        onodeSliderList.append(Scale(frame_backqueryNN, from_=1, to=99, length=minWidth, tickinterval=9, orient=HORIZONTAL))
        # show slider
        onodeSliderList[onode].grid(row=onode, column=1)
    
    # create the button to compute
    button_Compute_Backquery = Button(frame_backqueryNN, padx=20, text="compute the backquery", command = lambda: compute_Backquery())
    # add the button to the grid
    button_Compute_Backquery.grid(row=0, column=3)
        
    # create the Frame to hold the picture
    backquery_Picture_Frame = Frame(frame_backqueryNN)
    # add the Frame to the grid
    backquery_Picture_Frame.grid(row=1, column=3, padx=40, rowspan=neuralNetwork.onodes-1)

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
    #fig = matplotlib.pyplot.figure(figsize=(5,5))
    #canvas = FigureCanvasTkAgg(fig, backquery_Picture_Frame)
    #canvas.draw()
    #canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)    

####################### for other #######################################

def reloadForNewNN():
    # reload the name to the new one
    chose_NN_Current.configure(text="current NN  ->  nodes: %s   learningrate: %s   trainingepochs: %s   performance: %s   "%(neuralNetwork.hnodes, neuralNetwork.lr, neuralNetwork.epochs, neuralNetwork.performance))
    # reload the backquerry sliders
    init_BackqueryNN_Tab()

































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
































# init all the stuff
getAllNN()
displayNN(numberNeuralNetworkList)

# start the GUI
root.mainloop()

