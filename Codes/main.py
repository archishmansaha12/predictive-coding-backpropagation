# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:13:23 2021

@author: archi
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import *
from utils import *
from neural_layers import *
from cnn import *

if __name__ == '__main__':
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #parser = argparse.ArgumentParser()
    print("Initialized")
    #parsing arguments
    #parser.add_argument("--savedir",type=str,default="savedir")
    batch_size = 64
    learning_rate = 0.0005
    N_epochs = 100
    save_every = 1
    #parser.add_argument("--print_every",type=int,default=10)
    #parser.add_argument("--old_savedir",type=str,default="None")
    n_inference_steps =100
    inference_learning_rate =0.1
    network_type = "pc"        ## for predictive coding
    #network_type = "backprop"  ## for backpropagation
    #dataset_name = "cifar100"
    dataset_name = "cifar"
    loss_fn ="mse"

    #create folders
    #if args.savedir != "":
    #    subprocess.call(["mkdir","-p",str(args.savedir)])
    #if args.logdir != "":
    #    subprocess.call(["mkdir","-p",str(args.logdir)])
    #print("folders created")
    dataset,testset = get_cnn_dataset(dataset_name,batch_size)
    loss_fn, loss_fn_deriv = parse_loss_function(loss_fn)

    if dataset_name == "cifar100":
        output_size=100
    elif dataset_name == "cifar":
        output_size = 10
    
    ### defines layers of neural network formed
    
    l1 = ConvLayer(32,3,6,64,5,learning_rate,relu,relu_deriv,device=DEVICE)
    l2 = MaxPool(2,device=DEVICE)
    l3 = ConvLayer(14,6,16,64,5,learning_rate,relu,relu_deriv,device=DEVICE)
    l4 = ProjectionLayer((64,16,10,10),200,relu,relu_deriv,learning_rate,device=DEVICE)
    l5 = FCLayer(200,150,64,learning_rate,relu,relu_deriv,device=DEVICE)
    #if args.loss_fn == "crossentropy":
    #  l6 = FCLayer(150,output_size,64,learning_rate,softmax,linear_deriv,device=DEVICE)
    #else:
    l6 = FCLayer(150,output_size,64,learning_rate,linear,linear_deriv,device=DEVICE)
    
    layers = [l1,l2,l3,l4,l5,l6]
    
    if network_type == "pc":
        net = PCNet(layers, n_inference_steps, inference_learning_rate, loss_fn, loss_fn_deriv, device=DEVICE)
    elif network_type == "backprop":
        net = Backprop_CNN(layers,loss_fn = loss_fn,loss_fn_deriv = loss_fn_deriv)
    else:
        raise Exception("Network type not recognised: must be one of 'backprop', 'pc'")
    
    losses, accuracies, test_accuracies = net.train(dataset[0:-2],testset[0:-2],N_epochs,n_inference_steps)
        
    epochs = np.linspace(1,N_epochs,N_epochs)
    
    plt.plot(epochs,accuracies, label="Backpropagation on training dataset")
    plt.plot(epochs,test_accuracies, label="Backpropagation on test dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for backpropagation on CIFAR-10 dataset")
    plt.legend()
    plt.show()
    