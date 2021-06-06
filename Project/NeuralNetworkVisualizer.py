"""Neural network visualizer
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
        [5] create a directory for the logs
            https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
        [6] convert the canvas to a jpg file (temporary)
            https://stackoverflow.com/questions/9886274/how-can-i-convert-canvas-content-to-an-image

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
from logging import log


try:
    import glob
    import numpy
    import os
    import pickle
    import dill
    import matplotlib.pyplot
    import cv2
    import logging
    import datetime
    import math
    from PIL import Image
    from PIL import ImageDraw

    from tkinter import *
    from tkinter import ttk
    from tkinter import filedialog
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from PIL import ImageTk, Image
    from NumberNN import *
    from pathlib import Path
except ImportError as e:
    logging.error("Import: The Programm was unable to import all modules\n%s"%(e))






class neuralNetworkVisualizer:
    """ neuralNetworkVisualizer

        * this class displays the GUI in which
        * the neural network can be chosen
        * pictures files can be queried
        * pictures can be drawn and queried
        * the neural network can be backqueried

        param:
            root(Tkinter):                          window and container for the GUI

        attributes:
            notebook(Notebook):                     container for the tabs in the GUI
            frame_choose_trained_nn(Frame):         tab nr.1
            frame_query_picture(Frame):             tab nr.2
            frame_draw_picture(Frame):              tab nr.3
            frame_backquery_nn(Frame):              tab nr.4
            chose_nn_current_label(Label):          label to display the current neural network
            choose_nn_available_listbox(Listbox):   listbox to display all available neural networks
            selected_picture_label(Label):          label to display the select picture from the filedialog
            query_picture_result_frame(Frame):      frame to display the result of the neural network (tab nr.2)
            draw_picture_result_frame(Frame):       frame to display the result of the neural network (tab nr.3)            
            canvas(Canvas):                         canvas to draw an own image
            backquery_slider_frame(Frame):          frame for the sliders in tab nr.4
            backquery_picture_frame(Frame):         frame for the picture in tab nr.4
            self.temporary_drawn_image(Image):      temporary image that can be queried through the neural network
            self.pil_canvas(ImageDraw):             drawing on the temporary image

            file_dir(Path):                         relative path of the Project folder
            trained_nn_path_list(Path):             path of for the different saved neural networks 
            logging_file_name(Path):                path of the logging file

            neural_network_list(list[class]):       list for the loaded neural networks in
            neural_network(class):                  currently acitve neural network 
            current_active_nn(int):                 determins which neural network is active 
            onode_slider_list(list[Slider]):        slider for backquery 
            picture_filename(path):                 filepath to query a picture 
            onnode_names(list[String]):             the "name" of the ouputNodes from the NN 
            color_fg(color):                        foreground color of the canvas 
            color_bg(color):                        background color of the canvas 
            canvas_old_x(int):                      x coordinate on the canvas 
            canvas_old_y(int):                      y coordinate on the canvas  
            penwidth(int):                          penwidth on the canvas 

        test:
            * Opens and displays correct GUI
            * loads the neural networks
    """
    logging.info("Class: neuralNetworkVisualizer")

    def __init__(self, root):

        # method parameters
        self.root = root

        # attributes
        self.notebook = None
        self.frame_choose_trained_nn = None
        self.frame_query_picture = None
        self.frame_draw_picture = None
        self.frame_backquery_nn = None
        self.chose_nn_current_label = None
        self.choose_nn_available_listbox = None
        self.selected_picture_label = None
        self.query_picture_result_frame = None
        self.draw_picture_result_frame = None
        self.canvas = None
        self.backquery_slider_frame = None
        self.backquery_picture_frame = None
        self.temporary_drawn_image = None
        self.pil_canvas = None

        self.file_dir = None
        self.trained_nn_path_list = None
        self.logging_file_name = None

        self.neural_network_list = [[],[]]
        self.neural_network = []
        self.current_active_nn = 0
        self.onode_slider_list=[]
        self.picture_filename = None
        self.onnode_names = [
                                ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
                                ["cat", "dog"]
                            ]

        # vars for the canvas - drawing
        # source [2]
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.canvas_old_x = None
        self.canvas_old_y = None
        self.penwidth = 20

        self.create_paths()

        # logging
        # configure the logger
        logging.basicConfig(filename=self.logging_file_name, level=logging.DEBUG, force=TRUE)
        logging.warning("Function: __init__: logging file was created \npath: %s"%(self.logging_file_name))                
        logging.info("Function: __init__: all variables are initialized")


       
        # create the Tkinter window
        self.root.title("Neural Network Visualizer")
        self.root.geometry("1000x600")

        # build the GUI
        self.init_GUI_layout()
        self.init_GUI_tabs()
        self.init_GUI_tab_1()
        self.init_GUI_tab_2()
        self.init_GUI_tab_3()
        self.init_GUI_tab_4()
        logging.info("Function: __init__: GUI is build")






    ####################################################
    ##                       GUI                      ##
    ####################################################


    def init_GUI_layout(self):
        """init_GUI_layout
            * creates the basic GUI layout 
            * a frame at the top with the buttons
            * and a frame beneath with the content

            param:
                None

            return:
                none

            test:
                * window size is correct
                * buttons exist
        """
        logging.info("GUI: initializing the window")

        # create a Frame for the buttons at the top
        root_button_frame = Frame(self.root, padx=100, pady=5)
        # create the NN buttons at the top
        button_choose_nn_number = Button(root_button_frame, text="Neural Network Numbers", command=self.open_nn_number)
        button_choose_nn_catdog = Button(root_button_frame, text="Neural Network Cat Or Dog", command=self.open_nn_catdog)
        # add the NN buttons to the grid
        button_choose_nn_number.pack(side=LEFT)
        button_choose_nn_catdog.pack(side=RIGHT)
        # add the Frame
        root_button_frame.pack(fill=X)

        # create a Frame for the notebook
        root_notebook_frame = Frame(self.root)
        # create a label for the notebook
        notebook_label = Label(root_notebook_frame)
        # place the label for the notebook
        notebook_label.pack(fill=BOTH, expand=TRUE)
        # add the Frame
        root_notebook_frame.pack(fill=BOTH, expand=TRUE)

        # create the notebook for the tabs
        self.notebook = ttk.Notebook(notebook_label)
        self.notebook.pack(fill=BOTH, expand=TRUE)





    def init_GUI_tabs(self):
        """init_GUI_tabs
            * creates the tabs for the different actions that can be performed

            param:
                None

            return:
                none

            test:
                * all tabs are created
                * names are correct
        """
        logging.info("GUI: creating tabs in GUI")

        # create the tabs
        self.frame_choose_trained_nn = Frame(self.notebook)
        self.frame_query_picture = Frame(self.notebook, padx=20, pady=20)
        self.frame_draw_picture = Frame(self.notebook)
        self.frame_backquery_nn = Frame(self.notebook, padx=20, pady=20)

        self.frame_choose_trained_nn.pack(fill=BOTH, expand=TRUE)
        self.frame_query_picture.pack(fill=BOTH, expand=TRUE)
        self.frame_draw_picture.pack(fill=BOTH, expand=TRUE)
        self.frame_backquery_nn.pack(fill=BOTH, expand=TRUE)

        # add the tabs to the notebook
        self.notebook.add(self.frame_choose_trained_nn, text="choose a trained nn")
        self.notebook.add(self.frame_query_picture, text="query a picture")
        self.notebook.add(self.frame_draw_picture, text="draw a picture")
        self.notebook.add(self.frame_backquery_nn, text="backquery the nn")





    def init_GUI_tab_1(self):
        """init_GUI_tab_1
            * creates the content of the first tab
            * - label of the current neural network at the top
            * - list of available neural networks
            * - button to choose a neural network

            param:
                None

            return:
                none

            test:
                * all tabs are created
                * names are correct
        """
        logging.info("GUI: initialize tab 1: choose a trained nn")

        # add a label to self.frame_choose_trained_nn for the current NN
        self.chose_nn_current_label = Label(self.frame_choose_trained_nn, padx=10, pady=20, text="no nn chosen currently")
        self.chose_nn_current_label.pack(fill=X)

        # add a Frame to frame_choose_trained_nn for available NN
        choose_nn_available_frame = Frame(self.frame_choose_trained_nn, bg="green")
        choose_nn_available_frame.pack(fill=X)
        # create a scrollbar for the available NN
        choose_nn_available_scrollbar = Scrollbar(choose_nn_available_frame)
        # create a Listbox for the available NN
        self.choose_nn_available_listbox = Listbox(choose_nn_available_frame,height=25, yscrollcommand=choose_nn_available_scrollbar.set)
        # configure the scrollbar
        choose_nn_available_scrollbar.config(command=self.choose_nn_available_listbox.yview)
        # add scrollbar 
        choose_nn_available_scrollbar.pack(side=RIGHT, fill=Y)
        # add Listbox
        self.choose_nn_available_listbox.pack(fill=BOTH, expand=TRUE)

        # create a Button to select a NN
        choose_nn_available_button = Button(self.frame_choose_trained_nn, text="select nn", padx=20, pady=5, command=self.load_nn)
        # add the button
        choose_nn_available_button.pack(side=TOP, fill=X)





    def init_GUI_tab_2(self):
        """init_GUI_tab_2
            * creates the content of the second tab
            * - button to open a filedialog and choose a picture
            * - label with the path of the picture
            * - button to query the picture
            * - frame for the results of the neural network

            param:
                None

            return:
                none

            test:
                * all tabs are created
                * names are correct
        """
        logging.info("GUI: initialize tab 2: query a picture")

        # create a Button to select a file
        select_picture_dialog_button = Button(self.frame_query_picture, text="select a picture", padx=20, pady=5, command=self.select_picture)
        # add the button
        select_picture_dialog_button.pack(side=TOP)

        # add a label with the picture path
        self.selected_picture_label = Label(self.frame_query_picture, padx=10, pady=20, text="no picture chosen currently")
        # add the label
        self.selected_picture_label.pack(side=TOP, fill=X)

        # create a button to query the picture
        query_picture_button = Button(self.frame_query_picture, text="query the picture", padx=20, pady=5, command=self.query_picture)
        # add the button
        query_picture_button.pack(side=TOP)

        # create a frame to display the result of the neural network
        self.query_picture_result_frame = Frame(self.frame_query_picture, padx=20, pady=30)
        # add the frame
        self.query_picture_result_frame.pack(fill=X)





    def init_GUI_tab_3(self):
        """init_GUI_tab_3
            * creates the content of the third tab
            * - canvas to draw a picture
            * - button to clear the canvas
            * - button to query the picture
            * - frame for the results of the neural network

            param:
                None

            return:
                none

            test:
                * all tabs are created
                * names are correct
        """
        logging.info("GUI: initialize tab 3: draw a picture")

        # create a frame for the buttons of the canvas
        draw_buttons_frame = Frame(self.frame_draw_picture, padx = 5, pady = 5)
        # create a Button to clear the canvas
        draw_picture_clear_button = Button(draw_buttons_frame, text="clear the canvas", padx=20, pady=5, command=self.clear)
        draw_picture_clear_button.pack(side=TOP)
        # create a Button to query the drawn image
        query_drawn_picture_button = Button(draw_buttons_frame, text="query the image", padx=20, pady=5, command=self.query_drawn_image)
        query_drawn_picture_button.pack(side=TOP)
        # add the frame
        draw_buttons_frame.pack(side=LEFT)

        # create the canvas to draw on
        self.canvas = Canvas(self.frame_draw_picture,width=300,height=300,bg=self.color_bg)

        # PIL create an empty image and draw object to draw on   memory only, not visible
        # source [6]
        self.temporary_drawn_image = Image.new("RGB", (300, 300), (255, 255, 255))
        self.pil_canvas = ImageDraw.Draw(self.temporary_drawn_image)

        # add the canvas
        self.canvas.pack(side=LEFT)
        # adding events to the canvas to draw on
        self.canvas.bind('<B1-Motion>',self.paint)
        #drawing the line 
        self.canvas.bind('<ButtonRelease-1>',self.reset)

        # create a frame to display the result of the neural network
        self.draw_picture_result_frame = Frame(self.frame_draw_picture, padx=20, pady=30)
        # add the frame
        self.draw_picture_result_frame.pack(side=RIGHT)





    def init_GUI_tab_4(self):
        """init_GUI_tab_4
            * creates the content of the fourth tab
            * - sliders to regulate the output nodes
            * - button to backquery the output nodes
            * - frame for the results of the neural network

            param:
                None

            return:
                none

            test:
                * all tabs are created
                * names are correct
        """
        logging.info("GUI: initialize tab 4: backquery the nn")

        # create a frame for the ouput node slider
        self.backquery_slider_frame = Frame(self.frame_backquery_nn, padx = 5, pady = 5)
        # add the frame
        self.backquery_slider_frame.grid(row=0, column=0)

        # create the button to compute
        button_compute_backquery = Button(self.frame_backquery_nn, padx=20, pady=5, text="compute the backquery", command = self.compute_backquery)
        # add the button to the grid
        button_compute_backquery.grid(row=1, column=0)
            
        # create the Frame to hold the picture
        self.backquery_picture_frame = Frame(self.frame_backquery_nn)
        # add the Frame to the grid
        self.backquery_picture_frame.grid(row=0, column=1, padx=40, rowspan=2)










    ##########################################################
    ##                       Functions                      ##
    ##########################################################


    ####################### for the nn buttons ###################################


    def open_nn_number(self):
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

        self.current_active_nn = 0
        logging.info("Function: open_nn_number: current neural network is %s"%(self.current_active_nn))

        # load the data into the GUI
        self.load_data()
        return





    def open_nn_catdog(self):
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

        self.current_active_nn = 1
        logging.info("Function: open_nn_catdog: current neural network is %s"%(self.current_active_nn))

        # load the data into the GUI
        self.load_data()

        return




    ####################### for frame_choose_trained_nn tab ########################


    def get_all_nn(self):
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

        # tab 1
        # clear old List of these neural networks
        self.neural_network_list[self.current_active_nn] = []        
        # reset the nn label
        self.chose_nn_current_label.configure(text="no nn chosen currently")
        
        
        # tab 2
        # clear the path of the picture
        self.picture_filename = None
        # clear the path on the GUI
        self.selected_picture_label.configure(text="no picture chosen currently")
        # clear the frame that displays the output nodes in the frame_query_picture
        self.clear_frame(self.query_picture_result_frame)
        
        # tab 3
        # clear the canvas
        self.clear()
        # clear the frame that displays the output nodes in the frame_query_picture
        self.clear_frame(self.draw_picture_result_frame)

        # tab 4
        # clear the frame that displays the sliders
        self.clear_frame(self.backquery_slider_frame)
        # clear the frame that holds the computed picture
        self.clear_frame(self.backquery_picture_frame)

        # for every numberNN file in the folder
        for trained_nn_file in glob.glob(self.trained_nn_path_list[self.current_active_nn]):
            logging.info("get_all_nn load file %s"%(trained_nn_file))
            # open the file and save the object 
            with open(trained_nn_file, 'rb') as input:
                # append it to the list of NNs
                self.neural_network_list[self.current_active_nn].append(dill.load(input))





    def display_nn(self, nn_list):
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
        self.choose_nn_available_listbox.delete(0,END)

        # iterate through the array and display the NNs
        for nn in nn_list:
            # display text on the List like (nodes: 100   learningrate: 0.2   trainingepochs: 2   performance: 0.98   )
            display_text_choose_nn = "nodes: %s   learningrate: %s   trainingepochs: %s   performance: %s   "%(nn.hnodes, nn.lr, nn.epochs, nn.performance)
            # add the text to the listbox
            self.choose_nn_available_listbox.insert(END, display_text_choose_nn)





    def load_nn(self):
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

        # get the index of the select List entry
        load_index = self.choose_nn_available_listbox.curselection()[0]
        # the index is retrieved and the NN gets chosen out of the current nn_list
        self.neural_network = self.neural_network_list[self.current_active_nn][load_index]
        # reload 
        self.reload_for_new_nn()





    ####################### for frame_query_picture tab ########################


    def select_picture(self):
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

        # select a file path over dialog
        self.picture_filename = filedialog.askopenfilename(initialdir=self.file_dir, title="select an image to query", filetypes=(("png","*.png"),("jpg","*.jpg")))
        # print the path on the GUI
        self.selected_picture_label.configure(text="current path  ->  " + self.picture_filename)
        # add image to the GUI
        #selected_picture = ImageTk.PhotoImage(Image.open(self.picture_filename))
        #display_selected_picture_label = Label(frame_query_picture, image=selected_picture)
        #display_selected_picture_label.pack()





    def query_picture(self):
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
        imageToQuery = self.prepare_picture_for_nn(self.picture_filename)
        # query the picture
        resultList = self.neural_network.query(imageToQuery)
        # show results
        self.show_query_result(resultList, self.query_picture_result_frame)





    def prepare_picture_for_nn(self, picturePath):
        """prepare_picture_for_nn
            * picture is converted to one channel 
            * array of the value gets flattened to 1 dimension
            * values are transformed from 0-255 to 0.01-1.0

            param:
                picturePath(path): path of the picture to be prepared

            return:
                img_data(list[float]): 1D list of the grey values from the picture

            test:
                * convertion to grey is correct
                * transformation from 0-255 to 0.01-1.0 is correct 
        """
        logging.info("Function: prepare_picture_for_nn")

        # define the image size (only the square of the axis is given)
        img_size = int(math.sqrt(self.neural_network.inodes))
        # get image from path
        originalImage = cv2.imread(picturePath)
        # resize the image with openCv2
        resizedImage = cv2.resize(originalImage, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        # gray the image
        resizedGrayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
        # load image data into an array
        img_array = resizedGrayImage.flatten()
        # reshape from resized to input_nodes value, invert values
        img_data  = 255.0 - img_array.reshape(self.neural_network.inodes)
        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01
        # return the changed image data
        return img_data





    def show_query_result(self, resList, frame):
        """show_query_result
            * display the outputnode values on the screen
            * the highest one is chosen as guess of the neural network

            param:
                resList(list[float]):   array of the outputnode values of the neural network
                frame(Frame):           the frame on which the results should be displayed

            return:
                none

            test:
                * correct result is chosen
                * all outputnodes are displayed
        """
        logging.info("Function: show_query_result")

        # clear the old results
        self.clear_frame(frame)
        
        # the index of the highest value corresponds to the label
        result = numpy.argmax(resList)
        # add label with the result
        result_label = Label(frame, pady=20, text="neural Network thinks it's a %s"%(self.onnode_names[self.current_active_nn][result]))
        # show label
        result_label.pack(fill=X)

        # add for every calculated onode a value
        for index, result in enumerate(resList):
            # add label with node and result
            onoderesult_label = Label(frame, text="output node %s:   %.3f%%"%(self.onnode_names[self.current_active_nn][index], 100*result))
            # show label
            onoderesult_label.pack(fill=X)
        




    ####################### for frame_draw_picture tab #########################


    def paint(self,e):
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
        if self.canvas_old_x and self.canvas_old_y:
            self.canvas.create_line(self.canvas_old_x,self.canvas_old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=TRUE)
            # PIL can not draw a line with pen width - drawing a rectangle
            half = int(self.penwidth/2)
            self.pil_canvas.rectangle([self.canvas_old_x-half,self.canvas_old_y-half,e.x+half,e.y+half], self.color_fg)

        self.canvas_old_x = e.x
        self.canvas_old_y = e.y





    def reset(self,e):
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
        self.canvas_old_x = None
        self.canvas_old_y = None 





    def clear(self):
        """clear
            * clear the canvas

            param:
                none

            return:
                none

            test:
                * canvas is empty
                * 
        """
        logging.info("Function: clear")

        self.canvas.delete(ALL)
        self.pil_canvas.rectangle([0,0,300,300], self.color_bg)






    def query_drawn_image(self):
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
        logging.info("Function: query_drawn_image")
        
        # save image from canvas temporary
        filename = "temporary_save_picture_drawn_on_canvas.jpg"
        self.temporary_drawn_image.save(filename)

        # prepare the picture
        imageToQuery = self.prepare_picture_for_nn(filename)
        # query the picture
        resultList = self.neural_network.query(imageToQuery)
        # show results
        self.show_query_result(resultList, self.draw_picture_result_frame)
        
        # remove temporary saved image
        os.remove(filename)

        return





    ####################### for frame_backquery_nn tab #########################


    def load_backquery_nn_tab(self):
        """load_backquery_nn_tab
            * displays the output nodes of the neural network as slider

            param:
                onodel_slider_list(list[Slider]): list of slider that define the value of the ouput nodes

            return:
                none

            test:
                * slider labels have the correct name
                * sliders have the correct values
        """
        logging.info("Function: load_backquery_nn_tab")

        # clear the old slider
        self.clear_frame(self.backquery_slider_frame)
        # empty the slider list array
        self.onode_slider_list=[]

        # iterate through the the output nodes of the nn
        for onode in range(self.neural_network.onodes):
            # add label to specifiy which output node is altered
            onode_slider_label = Label(self.backquery_slider_frame, text="output node %s"%(self.onnode_names[self.current_active_nn][onode]))
            # show label
            onode_slider_label.grid(row=onode, column=0)
            # create slider to alter the output node and add it to the array
            self.onode_slider_list.append(Scale(self.backquery_slider_frame, from_=1, to=99, length=200, orient=HORIZONTAL))
            # show slider
            self.onode_slider_list[onode].grid(row=onode, column=1)
        






    def compute_backquery(self):
        """compute_backquery
            * backquery the neural network and display the picture
            * 1. get the value of the output node sliders
            * 2. backquery those value through the neural network
            * 3. display the graph

            param:
                onodel_slider_list(list[Slider]): list of slider that define the value of the ouput nodes

            return:
                none

            test:
                * slider values are retrieved correct
                * image has the right dimension
        """
        logging.info("Function: compute_backquery")

        # clear the old picture
        self.clear_frame(self.backquery_picture_frame)

        # node value list
        output_node_value = []
        # get all values in order - save in the output value list
        for slider in self.onode_slider_list:
            # divide by 100 to get percentage
            output_node_value.append(slider.get()/100.0)

        # backquery and get the image data
        image_data = self.neural_network.backquery(output_node_value)
        # define the image size (only the square of the axis is given)
        img_size = int(math.sqrt(self.neural_network.inodes))
        # plot image data
        matplotlib.pyplot.imshow(image_data.reshape(img_size,img_size), cmap='Greys', interpolation='None')
        # show the image in extra window
        matplotlib.pyplot.show()

        ## not working //TODO
        # show the image in the GUI
        # https://stackoverflow.com/questions/19612419/updating-matplotlib-imshow-from-within-a-tkinter-gui
        fig = matplotlib.pyplot.figure(figsize=(4,4))
        canvas = FigureCanvasTkAgg(fig, self.backquery_picture_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)    





    ####################### general #######################################


    def load_data(self):
        """load_data
            * loads data in the GUI
            * all neural networks are loaded and displayed

            param:
                none

            return:
                none

            test:
                * all data is loaded
                * TODO
        """
        logging.info("Function: load_data")
        self.get_all_nn()
        self.display_nn(self.neural_network_list[self.current_active_nn])





    def reload_for_new_nn(self):
        """reload_for_new_nn
            * GUI gets reset before displaying another neural network

            param:
                onodel_slider_list(list[Slider]): list of slider that define the value of the ouput nodes

            return:
                none

            test:
                * slider values are retrieved correct
                * image has the right dimension
        """
        logging.info("Function: reload_for_new_nn")
        
        # reload the name to the new one
        self.chose_nn_current_label.configure(text="current nn  ->  nodes: %s   learningrate: %s   trainingepochs: %s   performance: %s   "%(self.neural_network.hnodes, self.neural_network.lr, self.neural_network.epochs, self.neural_network.performance))
        # clear the frame that displays the output nodes in the frame_query_picture
        self.clear_frame(self.query_picture_result_frame)
        # clear the canvas
        self.clear()
        # reload the backquerry sliders
        self.load_backquery_nn_tab()





    def clear_frame(self, frame):
        """clear_frame
            * clears a frame

            param:
                frame(Frame): the frame that is to be cleaned

            return:
                none

            test:
                * frame is clear
                * 
        """
        logging.info("Function: clear_frame")
        # source [4]
        for widget in frame.winfo_children():
            widget.destroy()





    def create_paths(self):
        """clear_frame
            * initialize the different paths the class needs

            param:
                none

            return:
                none

            test:
                * all path exist
                * log dir is created if missing
        """
        logging.info("Function: create_paths")

        try: 
            # relative path to files
            # source [3]
            self.file_dir = os.path.dirname(os.path.realpath('__file__'))

            # path of for the different saved neural networks 
            self.trained_nn_path_list =  [
                                        os.path.join(self.file_dir, 'Trained_NN/Numbers/numberNN*'),
                                        os.path.join(self.file_dir, 'Trained_NN/CatDog/catdogNN*')
                                    ] 
            # dir for the logging file
            logging_file_path = os.path.join(self.file_dir, '../../Logs')

            try: 
                # create dir if not exist
                # source [5]
                Path(logging_file_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error("Function: create_paths: unable create the dir \n%s"%(e))

            # path of the logging file
            self.logging_file_name = os.path.join(logging_file_path, 'log_neuralNetworkVisualizer%s.log'%(datetime.datetime.now().strftime("-%Y_%m_%d-%H_%M_%S")))
        
        except Exception as e:
            logging.error("Function: create_paths: unable to create the paths \n%s"%(e))
        


    









######################################################################
##                       Init and Loop the GUI                      ##
######################################################################




if __name__ == "__main__":
    root = Tk()
    window = neuralNetworkVisualizer(root)
    root.mainloop()