import numpy as np
from scipy.optimize import fmin
from math import log

# class Neural Network that builds the architecture and has NN functions
class NeuralNetwork(object):

    def print_all(self):
        print "*"*60
        print "weights _theta"
        print self.weights_theta
        print "activation node"
        print self.activation_nodes_list
        print "error delta"
        print self.error_delta
        print "error accumulator"
        print self.error_accumulator
        print "gradient partial deriv"
        print self.gradient_partial_deriv
        print "*" * 60

    def __init__(self, size):
        # architecture of the network
        self.size = size
        # no of layers in the network
        self.no_of_layers = len(size)
        # theta which we need to optimize is initialized
        self.weights_theta = []
        # activation notes
        self.activation_nodes_list = []
        # error term which is calculated in back propogation
        self.error_delta = []
        # this acumulates all the error terms of a training example set 1 to m.
        self.error_accumulator = []
        # once error_accumulator is done, we need to calculate the partial derivative to feed to optimization funciton
        self.gradient_partial_deriv = []
        # initialize the activation nodes (includes input layer  and output layer)
        # theta, error accumulator, gradient partial derivative are all of same size
        # error_delta does not include input layer
        for i,s in enumerate(size[:len(size)-1]):
            self.activation_nodes_list.append(np.zeros((size[i])))
            self.weights_theta.append(np.resize(np.random.standard_normal(size[i+1]*(size[i]+1)),(size[i+1],size[i]+1)))
            self.error_accumulator.append(
                np.resize(np.zeros(size[i + 1] * (size[i] + 1)), (size[i + 1], size[i] + 1)))
            self.gradient_partial_deriv.append(
                np.resize(np.zeros(size[i + 1] * (size[i] + 1)), (size[i + 1], size[i] + 1)))
            # we dont have error_delta for the first layer, only from second layer onwards
            self.error_delta.append(np.empty((size[i+1])))
        self.activation_nodes_list.append(np.zeros((size[-1])))

    # sigmoid function that calculates activation nodes value as well as the derivative
    def sigmoid(self, z, deriv = False):
        if deriv:
            return np.dot(z,(1-z))
        else:
            return 1 / (1 + np.exp(-z))

    # feed forward where the activation nodes values are calculated using a given theta
    def feed_forward(self, input_data):
        #print self.activation_nodes_list
        #print "-"*30
        self.activation_nodes_list[0] = input_data
        for i in range(1, self.no_of_layers):
            # adding the bias unit and then dot product
            # we are not calculating derivative
            self.activation_nodes_list[i] = self.sigmoid(np.dot(self.weights_theta[i-1], np.insert(self.activation_nodes_list[i-1], 0, 1)))
        return self.activation_nodes_list[-1]
        #print self.activation_nodes_list

    # back propagation to calculate error_delta terms for a given theta
    def back_propogation(self, output):
        #print "backpropagation"
        #print self.error_delta
        # initializing last layer error delta
        self.error_delta[-1] = self.activation_nodes_list[-1] - output
        if self.no_of_layers <= 2:
            return
        self.error_delta[- 2] = np.dot(self.weights_theta[- 1].T, self.error_delta[- 1]) * self.sigmoid(
            self.activation_nodes_list[- 2], True)
        # we start from layer -1 till second layer
        for i in range(self.no_of_layers - 2, 1, -1):
            # remove the first row
            temp = self.error_delta[i - 1][2:]
            self.error_delta[i - 2] = np.dot(self.weights_theta[i - 1].T, temp) * self.sigmoid(
                self.activation_nodes_list[i - 2], True)

    # once
    def accumulation_function(self):
        #print "accumulation after feedforward and backpropagation"
        #print self.error_accumulator
        for i in range(0, self.no_of_layers -1):
            # need to convert vectors to array and then transpose
            # need to insert bias term in the activation nodes list
            #print i
            #print self.error_accumulator[i]
            #print np.array([np.insert(self.activation_nodes_list[i], 0, 1)])
            if i == self.no_of_layers -2:
                # this is required for the last layer activation nodes are all included
                temp = np.transpose(np.array([self.error_delta[i]]))
            else:
                # this is required so hidden layer activation nodes bias term is not included
                temp = np.transpose(np.array([self.error_delta[i]]))[1:]
            #print temp
            self.error_accumulator[i] = self.error_accumulator[i] + \
                np.dot(temp, np.array([np.insert(self.activation_nodes_list[i], 0, 1)]))
        #print self.error_accumulator

    def partial_derivative(self, no_of_training_examples, nn_lambda):
        for i in range(0, self.no_of_layers - 1):
            temp = self.weights_theta[i]
            # this is to make partical derivative zero whenever j = 0
            temp[:, 0] = 0
            #print self.error_accumulator
            #print nn_lambda
            #print temp
            #print no_of_training_examples
            # 1.0 to make the result floating not int
            self.gradient_partial_deriv[i] = (1.0/no_of_training_examples) * (self.error_accumulator[i] + \
                                             nn_lambda * temp)
        #print self.gradient_partial_deriv

        # costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda );
        # [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    def cost_function(self, unrolled_theta_weights, input_layer_size, hidden_layer_size, output_layer_size, input_data, output_data, lmbda):
        # J_temp = 1/m * (-y1' * log(a3) - (1 - y1)' * log(1-a3));
        # J = sum(diag(J_temp));
        cost = 0
        for index,val in enumerate(input_data):
            print "HELLLO"
            print np.array([output_data[index]])
            print self.sigmoid(self.feed_forward(val))
            #sum_one_example = -np.dot(output_data[index], log(self.sigmoid(val))) - \
             #   np.dot((1- output_data[index]), log(1 - self.sigmoid(val)))
            #cost += sum_one_example
        print cost
        #cost = -1/len(input_data) * sum(cost)
        return cost


    #def

    def theta_optimization(self):
        pass