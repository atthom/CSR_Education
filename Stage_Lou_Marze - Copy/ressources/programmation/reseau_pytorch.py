import torch as th
from torch import nn
import numpy as np
n_input = 1
n_output = 2
# # Example data set
data = [[-2], [0], [2]]
classes = [1, 0, 1]
data = th.Tensor(data)
classes = th.LongTensor(classes)
network = lossfn = optimizer = None

# Test network output before learning
# sensors = th.Tensor([[0], [1], [2]])
# output = network(sensors)
# print(output)

# Learning
# nrepeat = 2000
# for i in range(nrepeat):
    



def init(n_input, n_output):
    """
    Perform any needed initialization of the supervised algorithm (e.g.
    create a neural network). You might need to use the *global* keyword.
    :param n_input: number of inputs
    :param n_output: number of possible categories for output
    """
    # Init neural network
    global network
    network = nn.Sequential(
    nn.Linear(n_input, 100), # on converti les ninput en 100 output samples par regression lineaire
    nn.LeakyReLU(),
    nn.Linear(100, n_output), #pareil 
    )
    
    print("first layer: weights = ", network[0].weight)
    print("first layer: bias = ", network[0].bias)

    # Init loss function
    global lossfn
    lossfn = nn.CrossEntropyLoss()
    pred = network(data)
    print("loss:", lossfn(pred, classes))
    # Init optimizer
    global optimizer
    optimizer = th.optim.SGD(network.parameters(), lr=1e-1) #tensor optimized with stochastic gradient descent
    pass


def learn(X_train, Y_train):
    """
    Perform one learning step
    :param X_train: list of all training data
    :param y_train: list of all training labels
    :return: loss
    """
    optimizer.zero_grad() # set the gradient of all optimized tensors to 0
    pred = network(X_train)
    loss = lossfn(pred, Y_train)
    # print("loss:", lossfn(pred, classes))
    loss.backward() # this computes the gradient, i.e. the derivatives
    optimizer.step() # perform a single optimization step
    return loss


def take_decision(sensor):   # sensor is a tensor I think
    """
    Compute the label of a new data point
    :param sensors: data point
    :return: computed label of the new data point
    """
    pred = network(sensor)
    loss_1 = lossfn(pred,0)
    loss_2 = lossfn(pred, 1)
    if loss_1 <=loss_2:
        return 0
    else:
        return 1

# # Test network output after learning
# pred = network(data)
# #print("loss after", nrepeat, "learning steps:", lossfn(pred, classes))
# print("first layer: weights = ", network[0].weight)
# print("first layer: bias = ", network[0].bias)
