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

import copy, math, time, sys
import dataset_loader
from nn_ops import *

#############################
### Preliminaries
#############################

# Retrieve the arguments from the command-line
args = parseArgs()

# Fix the seed for the random generator
np.random.seed(seed=0)

#############################
### Dataset Handling
#############################

### Load the dataset
train_set, valid_set, test_set = dataset_loader.load_mnist()

### Define the dataset variables
n_training = train_set[0].shape[0]
n_feature = train_set[0].shape[1]
n_label = np.max(train_set[1])+1

#############################
### Neural Network parameters
#############################

### Activation function
act_func_name = args.act_func

### Network Architecture
nn_arch = np.array([n_feature] + args.arch + [n_label])

### Create the neural network
W,B,act_func,nb_params = initNetwork(nn_arch,act_func_name)

### Deep copy of parameters for the adaptive rule 
pW = copy.deepcopy(W)
pB = copy.deepcopy(B)

#############################
### Optimization parameters
#############################
eta = args.eta
batch_size = args.batch_size
n_batch = int(math.ceil(float(n_training)/batch_size))
n_epoch = args.n_epoch

#############################
### Auxiliary variables
#############################
cumul_time = 0.

# Convert the labels to one-hot vector
one_hot = np.zeros((n_label,n_training))
one_hot[train_set[1],np.arange(n_training)]=1.

printDescription("BURN-IN period", eta, nn_arch, act_func_name, batch_size, nb_params)
print("epoch time(s) train_loss train_accuracy valid_loss valid_accuracy eta") 

#############################
### Learning process
#############################
for i in range(n_epoch):
    for j in range(n_batch):

        ### Mini-batch creation
        batch, one_hot_batch, mini_batch_size = getMiniBatch(j, batch_size, train_set, one_hot)

        prev_time = time.clock()

        ### Forward propagation
        Y,Yp = forward(act_func, W, B, batch)

        ### Compute the softmax
        out = softmax(Y[-1])

        ### Compute the gradient at the top layer
        derror = out-one_hot_batch

        ### Backpropagation
        gradB = backward(derror, W, Yp)

        ### Update the parameters
        W, B = update(eta, batch_size, W, B, gradB, Y)

        curr_time = time.clock()
        cumul_time += curr_time - prev_time

    ### Training accuracy
    train_loss, train_accuracy = computeLoss(W, B, train_set[0], train_set[1], act_func) 

    ### Valid accuracy
    valid_loss, valid_accuracy = computeLoss(W, B, valid_set[0], valid_set[1], act_func) 

    result_line = str(i) + " " + str(cumul_time) + " " + str(train_loss) + " " + str(train_accuracy) + " " + str(valid_loss) + " " + str(valid_accuracy) + " " + str(eta)

    print(result_line)
    sys.stdout.flush() # Force emptying the stdout buffer

print("Finite difference Bprop")
eps = 1e-6
threshold = 1e-4
for j in range(n_training):

    ### Mini-batch creation
    print("Checking for example " + str(j))
    batch, one_hot_batch, mini_batch_size = getMiniBatch(j, 1, train_set, one_hot)
    
    for k in range(len(W)):
        
        print("Checking for weights")
        for m in range(W[k].shape[0]):
            for n in range(W[k].shape[1]):

                #####################
                # TO BE COMPLETED
                assert(False),"change the parameter W[k][m,n] a little bit forward "
                #####################
                
                ### Forward propagation
                Y1,Yp1 = forward(act_func, W, B, batch)
                
                ### Compute the softmax
                out1 = softmax(Y1[-1])

                #####################
                # TO BE COMPLETED
                # 
                assert(False),"compute the negative-log likelihood loss1 based on the true label train_set[1][j]"
                #####################

                #####################
                # TO BE COMPLETED
                # 
                assert(False),"change the parameter W[k][m,n] a little bit backward"
                #####################                
                
                ### Forward propagation
                Y2,Yp2 = forward(act_func, W, B, batch)
                
                ### Compute the softmax
                out2 = softmax(Y2[-1])

                #####################
                # TO BE COMPLETED
                # 
                assert(False),"compute the negative-log likelihood loss2 based on the true label train_set[1][j]"
                #####################

                #####################
                # TO BE COMPLETED
                assert(False),"restore the parameter W[k][m,n] to the initial value"
                #####################                
                
                ### Forward propagation
                Y,Yp = forward(act_func, W, B, batch)
                
                ### Compute the softmax
                out = softmax(Y[-1])
                
                ### Compute the gradient at the top layer
                derror = out-one_hot_batch
                
                ### Backpropagation
                gradB = backward(derror, W, Yp)
                
                grad_w = gradB[k].dot(Y[k].T)
                
                estimated_gradient = (loss1-loss2)/(2.0*eps)
                if(np.abs(estimated_gradient)>1e-6):
                    rel_err = np.abs((grad_w[m,n]-estimated_gradient)/(estimated_gradient+grad_w[m,n]))
                    print(str(rel_err)+" should be lower than 1e-4")
                    assert(rel_err<threshold),"check failed: "+str(rel_err)+" is greater than " + str(threshold) + " at layer " + str(k) +" for weight (" + str(m) + "," + str(n) + ")"
                    
        print("Checking for bias")
        for n in range(B[k].shape[0]):

            #####################
            # TO BE COMPLETED
            assert(False),"change the parameter B[k][n] a little bit forward "
            #####################
            
            ### Forward propagation
            Y1,Yp1 = forward(act_func, W, B, batch)
            
            ### Compute the softmax
            out1 = softmax(Y1[-1])

            #####################
            # TO BE COMPLETED
            # 
            assert(False),"compute the negative-log likelihood loss1 based on the true label train_set[1][j]"
            #####################

            #####################
            # TO BE COMPLETED
            # 
            assert(False),"change the parameter B[k][n] a little bit backward"
            #####################                
            
            ### Forward propagation
            Y2,Yp2 = forward(act_func, W, B, batch)
            
            ### Compute the softmax
            out2 = softmax(Y2[-1])

            #####################
            # TO BE COMPLETED
            # 
            assert(False),"compute the negative-log likelihood loss2 based on the true label train_set[1][j]"
            #####################

            #####################
            # TO BE COMPLETED
            # 
            assert(False),"restore the parameter B[k][n] to the initial value"
            #####################            
            
            ### Forward propagation
            Y,Yp = forward(act_func, W, B, batch)
            
            ### Compute the softmax
            out = softmax(Y[-1])
            
            ### Compute the gradient at the top layer
            derror = out-one_hot_batch
            
            ### Backpropagation
            gradB = backward(derror, W, Yp)
            grad_b = np.sum(gradB[k],1).reshape(-1,1)
            
            estimated_gradient = (loss1-loss2)/(2.0*eps)
            if(np.abs(estimated_gradient)>1e-6):
                rel_err = np.abs((grad_b[n]-estimated_gradient)/(estimated_gradient+grad_b[n]))
                print(str(rel_err)+" should be lower than 1e-4")
                assert(rel_err<threshold),"check failed: "+str(rel_err)+" is greater than " + str(threshold) + " at layer " + str(k) +" for bias " + str(n)
                    
