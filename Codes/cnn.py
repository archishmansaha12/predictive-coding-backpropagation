# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:12:26 2021

@author: archi
"""

import numpy as np
import torch
from dataset import *
from utils import *
from neural_layers import *


class PCNet(object):
  def __init__(self, layers, n_inference_steps_train, inference_learning_rate,loss_fn, loss_fn_deriv,device='cpu',numerical_check=False):
    self.layers= layers
    self.n_inference_steps_train = n_inference_steps_train
    self.inference_learning_rate = inference_learning_rate
    self.device = device
    self.loss_fn = loss_fn
    self.loss_fn_deriv = loss_fn_deriv
    self.L = len(self.layers)
    self.outs = [[] for i in  range(self.L+1)]
    self.prediction_errors = [[] for i in range(self.L+1)]
    self.predictions = [[] for i in range(self.L+1)]
    self.mus = [[] for i in range(self.L+1)]
    self.numerical_check = numerical_check
    if self.numerical_check:
      print("Numerical Check Activated!")
      for l in self.layers:
        l.set_weight_parameters()

  def update_weights(self,print_weight_grads=False,get_errors=False):
    weight_diffs = []
    for (i,l) in enumerate(self.layers):
      if i !=1:
        if self.numerical_check:
            true_weight_grad = l.get_true_weight_grad().clone()
        dW = l.update_weights(self.prediction_errors[i+1],update_weights=True)
        true_dW = l.update_weights(self.predictions[i+1],update_weights=True)
        diff = torch.sum((dW -true_dW)**2).item()
        weight_diffs.append(diff)
        if print_weight_grads:
          print("weight grads : ", i)
          print("dW: ", dW*2)
          print("true diffs: ", true_dW * 2)
          if self.numerical_check:
            print("true weights ", true_weight_grad)
    return weight_diffs


  def forward(self,x):
    for i,l in enumerate(self.layers):
      x = l.forward(x)
    return x

  def no_grad_forward(self,x):
    with torch.no_grad():
      for i,l in enumerate(self.layers):
        x = l.forward(x)
      return x

  def infer(self, inp,label,n_inference_steps=None):
    self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
    with torch.no_grad():
      self.mus[0] = inp.clone()
      self.outs[0] = inp.clone()
      for i,l in enumerate(self.layers):
        #initialize mus with forward predictions
        self.mus[i+1] = l.forward(self.mus[i])
        self.outs[i+1] = self.mus[i+1].clone()
      self.mus[-1] = label.clone() #setup final label
      self.prediction_errors[-1] = -self.loss_fn_deriv(self.outs[-1], self.mus[-1])#self.mus[-1] - self.outs[-1] #setup final prediction errors
      self.predictions[-1] = self.prediction_errors[-1].clone()
      for n in range(self.n_inference_steps_train):
      #reversed inference
        for j in reversed(range(len(self.layers))):
          if j != 0:
            self.prediction_errors[j] = self.mus[j] - self.outs[j]
            self.predictions[j] = self.layers[j].backward(self.prediction_errors[j+1])
            dx_l = self.prediction_errors[j] - self.predictions[j]
            self.mus[j] -= self.inference_learning_rate * (2*dx_l)
      #update weights
      weight_diffs = self.update_weights()
      #get loss:
      L = self.loss_fn(self.outs[-1],self.mus[-1]).item()#torch.sum(self.prediction_errors[-1]**2).item()
      #get accuracy
      acc = accuracy(self.no_grad_forward(inp),label)
      return L,acc,weight_diffs

  def test_accuracy(self,testset):
    accs = []
    for i,(inp, label) in enumerate(testset):
        pred_y = self.no_grad_forward(inp.to(DEVICE))
        acc =accuracy(pred_y,onehot(label).to(DEVICE))
        accs.append(acc)
    return np.mean(np.array(accs)),accs

  

  def train(self, dataset,testset,n_epochs,n_inference_steps):
    with torch.no_grad():
      accuracies = []
      losses = []
      test_accuracies =[]
      for n in range(n_epochs):
        print("Epoch: ",n)
        losslist = []
        for (i,(inp,label)) in enumerate(dataset):
          out = self.forward(inp.to(DEVICE))
          if self.loss_fn != cross_entropy_loss:
            label = onehot(label).to(DEVICE)
          else:
            label = label.long().to(DEVICE)
          L, acc,weight_diffs = self.infer(inp.to(DEVICE),label)
          losslist.append(L)
        mean_accuracy_epoch, acclist = self.test_accuracy(dataset)
        accuracies.append(mean_accuracy_epoch)
        mean_loss_epoch = np.mean(np.array(losslist))
        losses.append(mean_loss_epoch)
        mean_test_acc_epoch, _ = self.test_accuracy(testset)
        test_accuracies.append(mean_test_acc_epoch)
        print("Accuracy Training: ", mean_accuracy_epoch)
        
    return losses, accuracies, test_accuracies


  #def load_model(self,old_savedir):
  #    for (i,l) in enumerate(self.layers):
  #        l.load_layer(old_savedir,i)


class Backprop_CNN(object):
  def __init__(self, layers,loss_fn,loss_fn_deriv):
    self.layers = layers
    self.xs = [[] for i in range(len(self.layers)+1)]
    self.e_ys = [[] for i in range(len(self.layers)+1)]
    self.loss_fn = loss_fn
    self.loss_fn_deriv = loss_fn_deriv
    for l in self.layers:
      l.set_weight_parameters()

  def forward(self, inp):
    self.xs[0] = inp
    for i,l in enumerate(self.layers):
      self.xs[i+1] = l.forward(self.xs[i])
    return self.xs[-1]

  def backward(self,e_y):
    self.e_ys[-1] = e_y
    for (i,l) in reversed(list(enumerate(self.layers))):
      self.e_ys[i] = l.backward(self.e_ys[i+1])
    return self.e_ys[0]

  def update_weights(self,print_weight_grads=False,update_weight=False,sign_reverse=False):
    for (i,l) in enumerate(self.layers):
      dW = l.update_weights(self.e_ys[i+1],update_weights=update_weight,sign_reverse=sign_reverse)
      #if print_weight_grads:
      #  print("weight grads : ", i)
      #  print("dW: ", dW*2)
      #  print("weight grad: ",l.get_true_weight_grad())

  #def load_model(old_savedir):
  #    for (i,l) in enumerate(self.layers):
  #        l.load_layer(old_savedir,i)

  def test_accuracy(self,testset):
    accs = []
    for i,(inp, label) in enumerate(testset):
        pred_y = self.forward(inp.to(DEVICE))
        acc =accuracy(pred_y,onehot(label).to(DEVICE))
        accs.append(acc)
    return np.mean(np.array(accs)),accs

  def train(self, dataset,testset,n_epochs,n_inference_steps):
    with torch.no_grad():
      accuracies = []
      losses = []
      test_accuracies =[]
      for n in range(n_epochs):
        print("Epoch: ",n)
        #print("Loss")
        losslist = []
        for (i,(inp,label)) in enumerate(dataset):
          out = self.forward(inp.to(DEVICE))
          if self.loss_fn != cross_entropy_loss:
            label = onehot(label).to(DEVICE)
          else:
            label = label.long().to(DEVICE)
          e_y = self.loss_fn_deriv(out, label)
          #e_y = out - label
          self.backward(e_y)
          self.update_weights(update_weight=True,sign_reverse=True)
          #loss = torch.sum(e_y**2).item()
          loss = self.loss_fn(out, label).item()
          losslist.append(loss)
        mean_accuracy_epoch, acclist = self.test_accuracy(dataset)
        accuracies.append(mean_accuracy_epoch)
        mean_loss_epoch = np.mean(np.array(losslist))
        losses.append(mean_loss_epoch)
        mean_test_acc_epoch, _ = self.test_accuracy(testset)
        test_accuracies.append(mean_test_acc_epoch)
        print("Accuracy: ", mean_accuracy_epoch)
        
    return losses, accuracies, test_accuracies

