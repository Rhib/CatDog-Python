"""Neural network lab
    *
    *

    param:
        Author:     Jakob Schmidt & Jonas Bihr
        Date:       06.06.21
        Version:    1.0.0 
        Licencse:   free

    sources:
        [1] neural network code from 
            "Neuronale Netze selbst programmieren"
            "ein verständlicher einstieg mit Python"
            von Tariq Rashid, O'Reilly Verlag
            license GPLv2 
        [2] canvas to paint on
            https://github.com/abhishek305/Tkinter-Paint-app-demo/blob/master/Paint.py
        [3] relative paths
            https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
        [4] delete all kids from parent frame    
            https://stackoverflow.com/questions/50656826/how-can-i-delete-the-content-in-a-tkinter-frame/50657381


    Bewertung
        Projektkriterien:
        Kernkriterien:
        (20%) - alle Funktionen und Module sowie Klassen müssen Docstrings nach z.B. Googole (Docstrings pep8) enthalten
        (10%) - alle Funktionen und Klassen müssen jeweils 2 Testbeschreibungen enthalten

        Sitekriterien:
        (10%) - Eigenleistung: geeignetes Logverfahren suchen und anwenden
        (20%) - Codequalität und Stil
        (20%) - Funktionalität (requirement Informationen --- welche Module, Frameworks, Versionen, OS ... )

        weitere Kriterien:
        - wir wollen am Ende kein kundenfähiges System (20%)
        - Programm Intiutivität für den Nutzer möglichst einfach
        - Besondere Bemühungen und Aufwand, Elemente (Sound, Grafikdateien, Gameplay,... )
        - pi mal Daumen 48 Stunden zur Orientierung

        Bonuspunkte: -- Copy+Paste vs. Eigenaufwand
"""

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
import logging
import datetime









##########################################################
##                       Variables                      ##
##########################################################
# relative path to files
# source [3]
file_dir = os.path.dirname(os.path.realpath('__file__'))
# path of for the different saved neural networks 
trained_nn_path_list =  [
                            os.path.join(file_dir, 'Trained_NN/Numbers/numberNN*'),
                            os.path.join(file_dir, 'Trained_NN/CatDog/catdogNN*')
                        ] 
# path for the logging file
logging_file_path = os.path.join(file_dir, 'Logs/log%s.txt'%(datetime.datetime.now().strftime("-%Y_%m_%d-%H_%M_%S")))


# logging
# configure the logger
logging.basicConfig(filename=logging_file_path, level=logging.DEBUG)
logging.debug("debug")
logging.info("info")
logging.warning("warning")
logging.error("error")
logging.info("starting")


# global vars
windowWidth = 1000 
minWidth = 200
windowHeight = 600
# list for the loaded neural networks
neural_network_list = []
# currently acitve NN
neural_network = []
# checks if the number or the catdog NN is active
current_active_nn = 0
# slider for backquery
onode_slider_list=[]
# filepath to query a picture
pictuer_filename = ""
# the "name" of the ouputNodes from the NN
onnode_names =  [
                    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
                    ["cat", "dog"]
                ]


# vars for the canvas - drawing
# source [2]
color_fg = 'black'
color_bg = 'white'
canvas_old_x = None
canvas_old_y = None
penwidth = 20










####################################################
##                       GUI                      ##
####################################################


####################### Initialize ########################
logging.info("GUI: initializing the GUI")

# initialize the GUI
root = Tk()
root.title("Neural Network Visualizer")
root.geometry("1000x600")


# create the NN buttons at the top
button_choose_nn_number = Button(root, text="Neural Network Numbers", command = lambda: open_nn_number())
button_choose_nn_catdog = Button(root, text="Neural Network Cat Or Dog", command=lambda: open_nn_catdog())

# add the NN buttons to the grid
button_choose_nn_number.grid(row=0, column=0, columnspan=2)
button_choose_nn_catdog.grid(row=0, column=2, columnspan=2)


# create a label for the notebook
notebook_label = Label(root)

# place the label for the notebook
notebook_label.grid(row=1, column=0, columnspan=4)

# create the notebook for the tabs
notebook = ttk.Notebook(notebook_label)
notebook.pack()


####################### Tabs ########################
logging.info("GUI: creating tabs in GUI")

# create the tabs
frame_choose_trained_nn = Frame(notebook, width=windowWidth, height=windowHeight, bg="")
frame_query_picture = Frame(notebook, width=windowWidth, height=windowHeight, padx=20, pady=20)
frame_draw_picture = Frame(notebook, width=windowWidth, height=windowHeight)
frame_backquery_nn = Frame(notebook, width=windowWidth, height=windowHeight, padx=20, pady=20)

frame_choose_trained_nn.pack(fill=BOTH, expand=TRUE)
frame_query_picture.pack(fill=BOTH, expand=TRUE)
frame_draw_picture.pack(fill=BOTH, expand=TRUE)
frame_backquery_nn.pack(fill=BOTH, expand=TRUE)

# add the tabs to the notebook
notebook.add(frame_choose_trained_nn, text="choose a trained NN")
notebook.add(frame_query_picture, text="query a picture")
notebook.add(frame_draw_picture, text="draw a picture")
notebook.add(frame_backquery_nn, text="backquery the NN")


####################### frame_choose_trained_nn Layout ########################
logging.info("GUI: initialize tab: choose a trained NN")

# add a label to frame_choose_trained_nn for the current NN
chose_nn_current = Label(frame_choose_trained_nn, padx=10, pady=20, text="no NN chosen currently")
chose_nn_current.pack(fill=X)

# add a Frame to frame_choose_trained_nn for available NN
choose_nn_available_frame = Frame(frame_choose_trained_nn, bg="green")
choose_nn_available_frame.pack(fill=X)
# create a scrollbar for the available NN
choose_nn_available_scrollbar = Scrollbar(choose_nn_available_frame)
# create a Listbox for the available NN
choose_nn_available_listbox = Listbox(choose_nn_available_frame,height=25, yscrollcommand=choose_nn_available_scrollbar.set)
# configure the scrollbar
choose_nn_available_scrollbar.config(command=choose_nn_available_listbox.yview)
# add scrollbar 
choose_nn_available_scrollbar.pack(side=RIGHT, fill=Y)
# add Listbox
choose_nn_available_listbox.pack(fill=BOTH, expand=TRUE)

# create a Button to select a NN
choose_nn_available_button = Button(frame_choose_trained_nn, text="select NN", padx=20, pady=5, command=lambda: load_nn())
# add the button
choose_nn_available_button.pack(side=TOP)



####################### frame_query_picture Layout ########################
logging.info("GUI: initialize tab: query a picture")

# create a Button to select a file
select_picture_dialog_button = Button(frame_query_picture, text="select a picture", padx=20, pady=5, command=lambda: select_picture())
# add the button
select_picture_dialog_button.pack(side=TOP)
# add a label with the picture path
selected_picture_label = Label(frame_query_picture, padx=10, pady=20, text="no picture chosen currently")
# add the label
selected_picture_label.pack(side=TOP, fill=X)

# create a Button to query the picture
query_picture_button = Button(frame_query_picture, text="query the picture", padx=20, pady=5, command=lambda: query_picture())
# add the button
query_picture_button.pack(side=TOP)

# add a Frame to frame_choose_trained_nn for available NN
query_picture_result_frame = Frame(frame_query_picture, padx=20, pady=30)
query_picture_result_frame.pack(fill=X)



####################### frame_draw_picture Layout ########################
logging.info("GUI: initialize tab: draw a picture")

# create a frame for the settings of the canvas
draw_settings_frame = Frame(frame_draw_picture, padx = 5, pady = 5)
# label for the pen size slider
#Label(draw_settings_frame, text='Pen Width:',font=('arial 18')).grid(row=0,column=0)
# add pen size slider
#draw_pen_size_slider = ttk.Scale(draw_settings_frame,from_= 5, to = 100,command=lambda: changeW(),orient=HORIZONTAL)
#draw_pen_size_slider.set(penwidth)
#draw_pen_size_slider.grid(row=0,column=1,ipadx=30)
# create a Button to clear the canvas
draw_picture_clear_button = Button(draw_settings_frame, text="clear the canvas", padx=20, pady=5, command=lambda: clear())
draw_picture_clear_button.pack(side=TOP)
# create a Button to query the drawn image
query_drawn_picture_button = Button(draw_settings_frame, text="query the image", padx=20, pady=5, command=lambda: query_drawn_image())
query_drawn_picture_button.pack(side=TOP)
# add the frame
draw_settings_frame.pack(side=LEFT)

c = Canvas(frame_draw_picture,width=300,height=300,bg=color_bg)
c.pack(side=LEFT)








##########################################################
##                       Functions                      ##
##########################################################


####################### for the NN buttons ###################################


def open_nn_number():
    """open_nn_number
        * changes settings to the neural network for numbers

        param:
            None

        return:
            none

        test:
            * variables are set correctly
            * 
    """
    logging.info("Function: open_nn_number")

    global current_active_nn
    current_active_nn = 0
    return





def open_nn_catdog():
    """open_nn_catdog
        * changes settings to the neural network for cat or dog

        param:
            None

        return:
            none

        test:
            * variables are set correctly
            * 
    """
    logging.info("Function: open_nn_catdog")

    global current_active_nn
    current_active_nn = 1
    return




####################### for frame_choose_trained_nn tab ########################


def get_all_nn():
    """get_all_nn
        * loads all neural networks, from the files
        * 1. go through every file with (*.nn) in the folder
        * 2. open the file
        * 3. load it into the list of available neural networks

        param:
            None

        return:
            none

        test:
            * all files are loaded
            * files are loaded without errors
    """
    logging.info("Function: get_all_nn")

    # for every numberNN file in the folder
    for trained_nn_file in glob.glob(trained_nn_path_list):
        print("get_all_nn current file: ",trained_nn_file)
        # open the file and save the object 
        with open(trained_nn_file, 'rb') as input:
            # append it to the list of NNs
            neural_network_list[current_active_nn].append(dill.load(input))





def display_nn(nn_list):
    """display_nn
        * display all loaded neural networks on the screen
        * go through the list of neural networks and add a description to a listbox

        param:
            nnList(list[neuralNetworks: class]): list of all loaded neuralNetworks

        return:
            none

        test:
            * the right data is retrieved
            * all neural networks are displayed
    """
    logging.info("Function: display_nn")

    # clear the old Listbox entries
    choose_nn_available_listbox.delete(0,END)

    # iterate through the array and display the NNs
    for nn in nn_list:
        # display text on the List like (nodes: 100   learningrate: 0.2   trainingepochs: 2   performance: 0.98   )
        display_text_choose_nn = "nodes: %s   learningrate: %s   trainingepochs: %s   performance: %s   "%(nn.hnodes, nn.lr, nn.epochs, nn.performance)
        # add the text to the listbox
        choose_nn_available_listbox.insert(END, display_text_choose_nn)





def load_nn():
    """load_nn
        * neural network gets set
        * after a neural network is selected from the listobx and the select button is clicked
        * the index of the selected listbox is retrieved 
        * and the neural network is selected from the array of neural networks

        param:
            neuralNetwork(neuralNetwork: class): the currently active neural network

        return:
            none

        test:
            * index is chosen correctly
            * neural network is class neuralNetwork
    """
    logging.info("Function: load_nn")

    global neural_network
    # get the index of the select List entry
    load_index = choose_nn_available_listbox.curselection()[0]
    # the index is retrieved and the NN gets chosen out of the current nn_list
    neural_network = neural_network_list[current_active_nn][load_index]
    # reload 
    reload_for_new_nn()





####################### for frame_query_picture tab ########################


def select_picture():
    """select_picture
        * after button click open a filedialog at the 'Project' folder
        * .png or .jpg files can be chosen
        * save the path and print it to the GUI

        param:
            picture_filename(path): path of the file the user selected to query by the neural network

        return:
            none

        test:
            * path is correct
            * dialog opens at the correct folder
    """
    logging.info("Function: select_picture")

    global pictuer_filename
    # select a file path over dialog
    pictuer_filename = filedialog.askopenfilename(initialdir=file_dir, title="select an image to query", filetypes=(("png","*.png"),("jpg","*.jpg")))
    # print the path on the GUI
    selected_picture_label.configure(text="current path  ->  " + pictuer_filename)
    # add image to the GUI
    #selected_picture = ImageTk.PhotoImage(Image.open(pictuer_filename))
    #display_selected_picture_label = Label(frame_query_picture, image=selected_picture)
    #display_selected_picture_label.pack()





def query_picture():
    """query_picture
        * the picture is prepared for the neural network
        * then queried through the neural network
        * and lastly the results are shown in the GUI

        param:
            none

        return:
            none

        test:
            * image is correct prepared for the query
            * calculated result is consistent
    """
    logging.info("Function: query_picture")

    # prepare the picture
    imageToQuery = prepare_picture_for_nn(pictuer_filename)
    # query the picture
    resultList = neural_network.query(imageToQuery)
    # show results
    show_query_result(resultList)





def prepare_picture_for_nn(picturePath):
    """prepare_picture_for_nn
        * picture is converted to one channel 
        * array of the value gets flattened to 1 dimension
        * values are transformed from 0-255 to 0.01-1.0

        param:
            resList(list[float]): array of the outputnode values of the neural network

        return:
            img_data(list[float]): 1D list of the grey values from the picture

        test:
            * convertion to grey is correct
            * transformation from 0-255 to 0.01-1.0 is correct 
    """
    logging.info("Function: prepare_picture_for_nn")

    # define the image size (only the square of the axis is given)
    img_size = int(math.sqrt(neural_network.inodes))
    # get image from path
    originalImage = cv2.imread(picturePath)
    # resize the image with openCv2
    resizedImage = cv2.resize(originalImage, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    # gray the image
    resizedGrayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
    # load image data into an array
    img_array = resizedGrayImage.flatten()
    # reshape from resized to input_nodes value, invert values
    img_data  = 255.0 - img_array.reshape(neural_network.inodes)
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    # return the changed image data
    return img_data





def show_query_result(resList):
    """show_query_result
        * display the outputnode values on the screen
        * the highest one is chosen as guess of the neural network

        param:
            resList(list[float]): array of the outputnode values of the neural network

        return:
            none

        test:
            * correct result is chosen
            * all outputnodes are displayed
    """
    logging.info("Function: show_query_result")

    # clear the old results
    # source [4]
    for widget in query_picture_result_frame.winfo_children():
        widget.destroy()
    
    # the index of the highest value corresponds to the label
    result = numpy.argmax(resList)
    # add label with the result
    result_label = Label(query_picture_result_frame, pady=20, text="neural Network thinks it's a %s"%(onnode_names[current_active_nn][result]))
    # show label
    result_label.pack(fill=X)

    # add for every calculated onode a value
    for index, result in enumerate(resList):
        # add label with node and result
        onoderesult_label = Label(query_picture_result_frame, text="output node %s:   %.3f%%"%(onnode_names[current_active_nn][index], 100*result))
        # show label
        onoderesult_label.pack(fill=X)
    




####################### for frame_draw_picture tab #########################


def paint(e):
    """paint
        * draws a line on the canvas 
        * at the current mouse position

        param:
            e(event): event if clicked on the canvas

        return:
            none

        test:
            * correct event is given
            * line is drawn correct
    """
    logging.info("Function: paint")

    # source [2]
    global canvas_old_x, canvas_old_y
    if canvas_old_x and canvas_old_y:
        c.create_line(canvas_old_x,canvas_old_y,e.x,e.y,width=penwidth,fill=color_fg,capstyle=ROUND,smooth=TRUE)

    canvas_old_x = e.x
    canvas_old_y = e.y





def reset(e):
    """reset
        * resets the mouse postion on canvas 
        * after releasing the button

        param:
            e(event): event if mouse button is released 

        return:
            none

        test:
            * correct event is given
            * variables are null
    """
    logging.info("Function: reset")

    # source [2]    
    global canvas_old_x, canvas_old_y
    canvas_old_x = None
    canvas_old_y = None 





def clear():
    """reset
        * resets the canvas

        param:
            none

        return:
            none

        test:
            * canvas is empty
            * 
    """
    logging.info("Function: reset")

    c.delete(ALL)






def query_drawn_image():
    """reset
        * drawn image is saved temporarly 
        * image is prepared for the network
        * image gets queried through the neural network 
        * results are displayed

        param:
            none

        return:
            none

        test:
            * 
            * 
        //TODO
    """
    logging.info("Function: reset")

    return





####################### for frame_backquery_nn tab #########################


def init_backquery_nn_tab():
    global onode_slider_list
    global backquery_picture_frame

    onode_slider_list=[]
    # iterate through the array and display the NNs
    for onode in range(neural_network.onodes):
        # add label to specifiy which output node is altered
        onode_slider_label = Label(frame_backquery_nn, text="output node %s"%(onnode_names[current_active_nn][onode]))
        # show label
        onode_slider_label.grid(row=onode, column=0)
        # create slider and add it to the array
        onode_slider_list.append(Scale(frame_backquery_nn, from_=1, to=99, length=minWidth, orient=HORIZONTAL))
        # show slider
        onode_slider_list[onode].grid(row=onode, column=1)
    
    # create the button to compute
    button_compute_backquery = Button(frame_backquery_nn, padx=20, pady=5, text="compute the backquery", command = lambda: compute_backquery())
    # add the button to the grid
    button_compute_backquery.grid(row=neural_network.onodes, column=0, columnspan=2)
    #button_compute_backquery.place(anchor="center")
        
    # create the Frame to hold the picture
    backquery_picture_frame = Frame(frame_backquery_nn, bg="black")
    # add the Frame to the grid
    backquery_picture_frame.grid(row=0, column=2, padx=40, rowspan=neural_network.onodes)





def compute_backquery():
    # clear the old picture
    # https://stackoverflow.com/questions/50656826/how-can-i-delete-the-content-in-a-tkinter-frame/50657381
    for widget in backquery_picture_frame.winfo_children():
        widget.destroy()

    # node value list
    output_node_falue = []

    # get all values in order - save in the output value list
    for slider in onode_slider_list:
        # divide by 100 to get percentage
        output_node_falue.append(slider.get()/100.0)
    print(output_node_falue)

    # backquery and get the image data
    image_data = neural_network.backquery(output_node_falue)
    # define the image size (only the square is given)
    img_size = int(math.sqrt(neural_network.inodes))
    # plot image data
    matplotlib.pyplot.imshow(image_data.reshape(img_size,img_size), cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
    # show the image in the GUI
    # https://stackoverflow.com/questions/19612419/updating-matplotlib-imshow-from-within-a-tkinter-gui
    fig = matplotlib.pyplot.figure(figsize=(4,4))
    canvas = FigureCanvasTkAgg(fig, backquery_picture_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)    





####################### for other #######################################


def reload_for_new_nn():
    # reload the name to the new one
    chose_nn_current.configure(text="current NN  ->  nodes: %s   learningrate: %s   trainingepochs: %s   performance: %s   "%(neural_network.hnodes, neural_network.lr, neural_network.epochs, neural_network.performance))
    # reload the backquerry sliders
    init_backquery_nn_tab()










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
get_all_nn()
display_nn(neural_network_list[current_active_nn])
# adding events to the canvas to draw on
c.bind('<B1-Motion>',paint)#drawing the line 
c.bind('<ButtonRelease-1>',reset)

# start the GUI
root.mainloop()