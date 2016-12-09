"""
MiniNN - Minimal Neural Network
This code is a straigthforward and minimal implementation 
of a multi-layer neural network for training on MNIST dataset.
It is mainly intended for educational and prototyping purpuses.
"""
__author__ = "Gaetan Marceau Caron (gaetan.marceau-caron@inria.fr)"
__copyright__ = "Copyright (C) 2015 Gaetan Marceau Caron"
__license__ = "CeCILL 2.1"
__version__ = "1.0"

import numpy as np
from scipy.stats import bernoulli
import argparse

#############################
### Core functions
#############################

def initNetwork(nn_arch, act_func_name):
    """
        Initialize the neural network weights, activation function and return the number of parameters
  
        :param nn_arch: the number of units per hidden layer 
        :param act_func_name: the activation function name (sigmoid, tanh or relu)
        :type nn_arch: list of int
        :type act_func_name: str
        :return W: a list of weights for each hidden layer
        :return B: a list of bias for each hidden layer
        :return act_func: the activation function
        :return nb_params: the number of parameters 
        :rtype W: list of ndarray
        :rtype B: list of ndarray
        :rtype act_func: function
        :rtype n_params: number of parameters
    """

    W,B = [],[]
    sigma = 1.0
    act_func = globals()[act_func_name] # Cast the string to a function
    nb_params = 0
    
    for i in range(np.size(nn_arch)-1):
        w = 0.01*np.ones((nn_arch[i+1],nn_arch[i]))
        W.append(w)
        b = np.sum(w,1).reshape(-1,1)/-2.0
        B.append(b)
        nb_params += nn_arch[i+1] * nn_arch[i] + nn_arch[i+1]

    return W,B,act_func,nb_params

def forward(act_func,W,B,X):
    """
        Perform the forward propagation

        :param act_func: the activation function 
        :param W: the weights
        :param B: the bias
        :param X: the input
        :type act_func: function
        :type W: list of ndarray
        :type B: list of ndarray
        :type X: ndarray
        :return Y: a list of activation values
        :return Yp: a list of the derivatives w.r.t. the pre-activation of the activation values
        :rtype Y: list of ndarray
        :rtype Yp: list of ndarray
    """
    Y,Yp = [np.transpose(X)],[]
    #####################
    # TO BE COMPLETED
    for i,w in enumerate(W):
        F, Fp = act_func(np.dot(w,Y[i]) + B[i])
        Y = Y.append(F)
        Yp= Yp.append(Fp)
    assert(False),"forward must be completed before testing"
    #####################    
    return Y,Yp

def backward(error,W,Yp):
    """
        Perform the backward propagation

        :param error: the gradient w.r.t. to the last layer 
        :param W: the weights
        :param Yp: the derivatives w.r.t. the pre-activation of the activation functions
        :type error: ndarray
        :type W: list of ndarray
        :type Yp: list of ndarray
        :return gradb: a list of gradient w.r.t. the pre-activation with this order [gradb_layer1, ..., error] 
        :rtype gradB: list of ndarray
    """

    gradB = [error]
    #####################
    # TO BE COMPLETED
    
    assert(False),"backward must be completed before testing"
    #####################
    return gradB

def update(eta, batch_size, W, B, gradB, Y):
    """
        Perform the update of the parameters

        :param eta: the step-size of the gradient descent 
        :param batch_size: number of examples in the batch (for normalizing)
        :param W: the weights
        :param B: the bias
        :param gradB: the gradient of the activations w.r.t. to the loss
        :param Y: the activation values
        :type eta: float
        :type batch_size: int
        :type W: list of ndarray
        :type B: list of ndarray
        :type gradB: list of ndarray
        :type Y: list of ndarray
        :return W: the weights updated 
        :return B: the bias updated 
        :rtype W: list of ndarray
        :rtype B: list of ndarray
    """
    #####################
    # TO BE COMPLETED
    #####################
    # Use updateParams(W[k],grad_w, eta) and updateParams(B[k],grad_b, eta)
    # grad_b should be a vector: object.reshape(-1,1) can be useful
    assert(False),"update must be completed before testing"
    return W,B

#############################
#############################

#############################
### Activation functions
#############################
def sigmoid(z):
    """
        Perform the sigmoid transformation to the pre-activation values

        :param z: the pre-activation values
        :param grad_flag: flag for computing the derivatives w.r.t. z
        :type z: ndarray
        :type grad_flag: boolean
        :return y: the activation values
        :return yp: the derivatives w.r.t. z
        :rtype y: ndarray
        :rtype yp: ndarray
    """
    #####################
    # TO BE COMPLETED
    #####################
    # compute the sigmoid y and its derivative yp
    y= 1/(1+math.exp(-z))
    yp= y*(1-y)
    assert(False),"sigmoid must be completed before testing"
    return y,yp

def softmax(z):
    """
        Perform the softmax transformation to the pre-activation values

        :param z: the pre-activation values
        :type z: ndarray
        :return: the activation values
        :rtype: ndarray
    """
    return np.exp(z-np.max(z,0))/np.sum(np.exp(z-np.max(z,0)),0.)
#############################

def updateParams(theta, dtheta, eta):
    """
        Perform the update of the parameters

        :param theta: the network parameters
        :param dtheta: the updates of the parameters
        :param eta: the step-size of the gradient descent 
        :type theta: ndarray
        :type dtheta: ndarray
        :type eta: float
        :return: the parameters updated 
        :rtype: ndarray
    """
    return theta - eta * dtheta
#############################

#############################
## Auxiliary functions 
#############################
def getMiniBatch(i, batch_size, train_set, one_hot):
    """
        Return a minibatch from the training set and the associated labels

        :param i: the identifier of the minibatch
        :param batch_size: the number of training examples
        :param train_set: the training set
        :param one_hot: the one-hot representation of the labels
        :type i: int
        :type batch_size: int
        :type train_set: ndarray
        :type ont_hot: ndarray
        :return: the minibatch of examples
        :return: the minibatch of labels
        :return: the number of examples in the minibatch
        :rtype: ndarray
        :rtype: ndarray
        :rtype: int
    """
    
    ### Mini-batch creation
    n_training = np.size(train_set[1])
    idx_begin = i * batch_size
    idx_end = min((i+1) * batch_size, n_training)
    mini_batch_size = idx_end - idx_begin

    batch = train_set[0][idx_begin:idx_end]
    one_hot_batch = one_hot[:,idx_begin:idx_end]

    return np.asfortranarray(batch), one_hot_batch, mini_batch_size

def computeLoss(W, B, batch, labels, act_func):
    """
        Compute the loss value of the current network on the full batch

        :param W: the weights
        :param B: the bias
        :param batch: the weights
        :param labels: the bias
        :param act_func: the weights
        :type W: ndarray
        :type B: ndarray
        :type batch: ndarray
        :type act_func: function
        :return loss: the negative log-likelihood
        :return accuracy: the ratio of examples that are well-classified
        :rtype: float
        :rtype: float
    """
    
    ### Forward propagation
    h = np.transpose(batch)
    for k in range(len(W)-1):
        h,hp = act_func(W[k].dot(h)+B[k])
    z = W[-1].dot(h)+B[-1]

    ### Compute the softmax
    out = softmax(z)
    pred = np.argmax(out,axis=0)
    fy = out[labels,np.arange(np.size(labels))]
    try:
        loss = np.sum(-1. * np.log(fy))/np.size(labels)
    except Exception:
        fy[fy<1e-4] = fy[fy<1e-4] + 1e-6
        loss = np.sum(-1. * np.log(fy))/np.size(labels)
    accuracy = np.sum(np.equal(pred,labels))/float(np.size(labels))        
    
    return loss,accuracy

def parseArgs():
    # Retrieve the arguments
    parser = argparse.ArgumentParser(description='MiniNN -- Minimalist code for Neural Network Learning')
    parser.add_argument('--arch', help='Architecture of the hidden layers', default=[100], nargs='+', type=int)
    parser.add_argument('--act_func', help='Activation function name', default="sigmoid", type=str)
    parser.add_argument('--batch_size', help='Minibatch size', default=500, type=int)
    parser.add_argument('--eta', help='Step-size for the optimization algorithm', default=1.0, type=float)
    parser.add_argument('--n_epoch', help='Number of epochs', default=5000, type=int)

    args = parser.parse_args()
    return args

def parseArgs_ipython(**kwarg):
    """ Adaptation de parseArgs() pour IPython """
    class Arguments():
        
        def __init__(self, arch = [100], act_func = "sigmoid", batch_size = 500, eta = .01, n_epoch = 100):
    
            self.arch = arch
            self.act_func = act_func
            self.batch_size = batch_size
            self.eta = eta
            self.n_epoch = n_epoch

    return Arguments(**kwarg)



#############################

#############################
## Printing function
#############################
def printDescription(algo_name, eta, nn_arch, act_func_name, minibatch_size, nb_param):
    print("Description of the experiment")
    print("----------")
    print("Learning algorithm: " + algo_name)
    print("Initial step-size: " + str(eta))
    print("Network Architecture: " + str(nn_arch))
    print("Number of parameters: " + str(nb_param))
    print("Minibatch size: " + str(minibatch_size))
    print("Activation: " +  act_func_name)
    print("----------")
#############################
