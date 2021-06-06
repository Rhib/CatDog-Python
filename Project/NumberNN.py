"""Number Neural Network
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

"""

import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# reading the NN 
import logging


#####################################################################
##                       Neural Network Class                      ##
#####################################################################

# neural network class definition
class neuralNetwork:
    """ neuralNetwork

        * this class is a neural network to recognize numbers from 0 to 9
        * it uses an activation function and an inversed activation function
        * testdata and trainingdata is used to train the neural network
        * the neural network can be queried and backqueried

        attributes:
            inodes(int):           set number of nodes in input layer 
            hnodes(int):           set number of nodes in hidden layer
            onodes(int):           set number of nodes in oputput layer
            lr(float):             learning rate in the neural network
            wih(list[int][float]): weight matrix input to hidden node
            who(list[int][float]): weight matrix hidden to output node
            performance(float):    indicator how good neural network works
            epochs(int):           reputation of training data

        test:
            * loads the neural networks
            * the neural network recognizes the correct number
    """

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden oputput layer
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
        # self.activation_function = lambda x: scipy.special.expit(x)
        # self.inverse_activation_function = lambda x: scipy.special.logit(x)
        
        # performance 
        self.performance = 0
        
        # epochs
        self.epochs = 0

        pass
    
    def activation_function(self, arr):
        """activation_function
            * function to activate the neural network 
            * sigmoid funtion is used to get values for weights between 0 and 1

            param:
                arr     array of numbers to be transformed with sigmoid (input of node)

            return:
                arr     array of sigmoid transformed values (output of node)

            test:
                * numbers are transformed correctly
                * return is correct format
        """
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
        """inverse_activation_function
            * function to use output values and the logarithm function to get input values

            param:
                arr     array of numbers to be transformed with log (output of node)

            return:
                arr     array of log transformed values (input of node)

            test:
                * numbers are transformed correctly
                * return is correct format
        """
        arrX = 0
        for x in arr:
            arrY = 0
            for y in x:
                arr[arrX][arrY] = scipy.special.logit(y)
                arrY += 1
                logging.debug("calculate inverse")
                pass
            arrX += 1
            pass
        return arr
    
    # train the neural network
    def train(self, inputs_list, targets_list):
        """train
            * query input list through neural network
            * output is compared to target_list
            * calculate difference back to weight function (who, wih)

            param:
                input_list(list(float))     data to be queried to neural network
                targets_list(list(float))   expected output of neural network
        
            return:
                none

            test:
                * node layers calculated correctly
                * weights are updated correctly
        """
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
        """query
            * layers calculate values
            * values are transfered between layers
            * output node is calculated

            param:
                inputs_list(list(float))     data to be queried to neural network   
            return:
                final_outputs(list(float))   calculated solution of the neural network

            test:
            * output is expected list
            * nodes are calculated correctly
        """
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
        """backquery
            * query the targets list backwards through neural network
            * result are input nodes

            param:
                targets_list(list(float))   expected output of neural network
            return:
                inputs(list(float))         expected input to get target list as result from neural network

            test:
            * input nodes calculated correctly
            * log function works correctly

        """
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
