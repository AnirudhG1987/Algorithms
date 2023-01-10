from numpy import genfromtxt
import numpy as np
import neuralnetwork



def tic_tac_toe_picture(list):
    for index, i in enumerate(list):
        if i == -1:
            print " ",
        elif i == 0:
            print "X",
        else:
            print "0",
        if index % 3 == 2:
            print


input_data = genfromtxt('data.txt', delimiter=',')
output_data = genfromtxt('output.txt', delimiter=',')

# this is to ensure the theta that is initialized is always the same set of data.
np.random.seed(1)

# creating the network architecture
nn = neuralnetwork.NeuralNetwork([9,10,9])
# nn.print_all()
for index,value in enumerate(input_data):
    #print "weights"
    #print nn.weights_theta
    nn.feed_forward(value)
    nn.back_propogation(output_data[index])
    nn.accumulation_function()
    #nn.print_all()

nn.partial_derivative(len(input_data), 0)
nn.print_all()

print nn.cost_function(input_data, output_data)

print "all done"