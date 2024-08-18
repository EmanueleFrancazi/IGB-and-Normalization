#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:32:03 2023

@author: emanuele
"""


import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import math
import wandb
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

#to augment cifar10
from PIL import Image, ImageEnhance, ImageOps
import random

from einops.layers.torch import Rearrange

#for sched
import warmup_scheduler

#from torchvision.transforms import v2

import torchvision.transforms.functional as TF

from collections import defaultdict
from collections import Counter

from contextlib import contextmanager

#manage the input of values from outside the script 
import argparse
import sys
import time

import copy


#for pre-trained architectures
import torchvision.models as models
import torchvision.transforms as transforms





#%%% function to set validation miles-stones

def ValidTimes(tot_epochs, N_Batches,N_steps):
    '''
    define the time time vector for the evaluation stops (stochastic case)
    NOTE: in case of RETRIEVE mode the new times will be equispaced but, in general, with a different spacing between consecutive times with respect to the old vector
    Returns
    -------
    Times : numpy array
        return the logarithmic equispaced steps for stochastic algorithms (as SGD).

    '''    
    
    
    MaxStep = tot_epochs*N_Batches #the last factor is due to the fact that in the PCNSGD we trash some batches (so the number of steps is slightly lower); 0.85 is roughtly set thinking that batch size, also in the unbalance case will be chosen to not esclude more than 15% of batches
    Times = np.logspace(0, np.log2(MaxStep),num=N_steps, base=2.) 
    Times = np.rint(Times).astype(int)   
    
    if N_steps>4:
        for ind in range(0,4): #put the initial times linear to store initial state
            Times[ind] = ind+1    
    
    for steps in range (0, N_steps-1):
        while Times[steps] >= Times[steps+1]:
            Times[steps+1] = Times[steps+1]+1


    return Times


#%%% CUSTOMIZE THE AXE ASSOCIATED TO LOGGED METRICS (WANDB)
def CustomizedX_Axis():
    """
    Set the default x axis to assign it on each group of logged measures

    Returns
    -------
    None.

    """   
    wandb.define_metric("Performance_measures/True_Steps_+_1")
    # set all other Performance_measures/ metrics to use this step
    wandb.define_metric("Performance_measures/*", step_metric="Performance_measures/True_Steps_+_1")    
    
    
    wandb.define_metric("Check/Epoch")
    # set all other Check/ metrics to use this step
    wandb.define_metric("Check/*", step_metric="Check/Epoch")         
    #wandb.define_metric("Check/Step_Norm", step_metric="Check/True_Steps_+_1")  
    #wandb.define_metric("GradientAngles/*", step_metric="GradientAngles/True_Steps_+_1")  


#%%% logging into wandb server 
def WandB_logs(Time, model):
    """
    here we log the relevant measures in wandb

    Parameters
    ----------
    TimeVec : TYPE
        DESCRIPTION.
    Comp : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    #TODO: for some reson the plot of classes valid accuracyis shifted forward in the wandb charts; this doesn't seems to happen for the training; fix this logging issue
    wandb.log({'Performance_measures/Valid_Accuracy': (np.array(model.ValAcc)[-1]),
               'Performance_measures/Valid_Loss': (np.array(model.ValLoss)[-1]),
               'Performance_measures/Valid_f0': (np.array(model.Valf0)[-1]),
               'Performance_measures/Valid_max_f': (np.array(model.ValMaxf)[-1]),
               'Performance_measures/Train_Accuracy': (np.array(model.TrainAcc)[-1]),
                'Performance_measures/Train_Loss': (np.array(model.TrainLoss)[-1]),
                'Performance_measures/Train_f0': (np.array(model.Trainf0)[-1]),
                'Performance_measures/Train_max_f': (np.array(model.TrainMaxf)[-1]),
                'Performance_measures/GradNorm': (np.array(model.GradNorm)[-1]),
                'Performance_measures/StepSize': (np.array(model.StepSize)[-1]),
               'Performance_measures/True_Steps_+_1': Time+1})

    for i in range(0, model.n_outputs):
        wandb.log({'Performance_measures/Training_Loss_Class_{}'.format(i): model.TrainPCLoss[-1][i], 
                   'Performance_measures/Valid_Loss_Class_{}'.format(i): model.ValPCLoss[-1][i],
                   'Performance_measures/Training_Acc_Class_{}'.format(i): model.TrainPCAcc[-1][i], 
                   'Performance_measures/Valid_Acc_Class_{}'.format(i): model.ValPCAcc[-1][i],
                   'Performance_measures/True_Steps_+_1': Time+1})
            
            

      

#%%% functions used during training



#TODO: IS MORE EFFICIENT TO DEFINE A CLASS OF MEASURES AND DEFINE ATTRIBUTE AT INIT TO USE EVERY TIME FOR EACH METHOD OF THE CLASS INSTEAD OF RECOMPUTE IT EVERY TIME (FOR EXAMPLE THE class_total)
def accuracy(outputs, labels, params):
    if params['Loss_function']=='Hinge':
        preds = torch.sign(outputs.squeeze()) #squeeze to have all the tensors in the same dimension (as it is for the labels)

        
    else:
        _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def guesses(outputs, num_data_points, params, n_batch):
    
    guesses = torch.zeros(len(params['label_list']))
    if params['Loss_function']=='Hinge':
        preds = torch.sign(outputs.squeeze())
    else:    
        _, preds = torch.max(outputs, dim=1)
    for i in params['label_list']:
        guesses[params['label_list'].index(i)]+= torch.sum(preds == i).item() 
    #if n_batch==0:
        #print('pred', preds)
        #print('label_list', params['label_list'])
        #print(guesses)
    return guesses/num_data_points

def per_class_acc(outputs, labels, num_classes, num_classes_datapoints, params):
    
    class_correct = torch.zeros(num_classes, device=params['device'])
    class_total = torch.zeros(num_classes, device=params['device'])

    if params['Loss_function']=='Hinge':
        preds = torch.sign(outputs.squeeze())
    else:      
        _, preds = torch.max(outputs, dim=1)
    corrects = preds == labels
    
    #print('les pred', preds)
    #print('les labels', labels)

    for i in params['label_list']:
        #print(params['label_list'],params['label_list'], i, params['label_list'].index(i))
        class_correct[params['label_list'].index(i)] += torch.sum((corrects) & (labels == i))
        class_total[params['label_list'].index(i)] = num_classes_datapoints[i]
    
    #print(class_correct, class_total)
    
    class_accuracy = class_correct / (class_total)  # Adding a small value to avoid division by zero
    
    return class_accuracy
    
def per_class_loss(losses, labels, num_classes, num_classes_datapoints, params):
    class_total=torch.tensor([num_classes_datapoints[i] for i in range(num_classes)], device=params['device'])
    
    #print('BEFORE',losses)
    
    losses = losses.to(params['device']).double()
    
    #print (losses)
    # List of custom labels
    label_list = params['label_list']
        
    # Convert labels to indices based on label_list
    label_indices = torch.tensor([label_list.index(label) for label in labels], device=params['device'])
    
    # Determine the number of classes (based on the custom label list)
    num_classes = len(label_list)
    
    #print(num_classes)

    # Create a mask where each row corresponds to a class and filters losses by class
    class_mask = torch.arange(num_classes, device=params['device']).unsqueeze(1) == label_indices.unsqueeze(0)
    

    
    class_mask_transposed = class_mask.T.double()
    
    #print('Funge? ',class_mask_transposed, losses)
    
    # Multiply the loss tensor by the class mask to zero out losses not belonging to the class
    class_loss = torch.matmul(losses.unsqueeze(0), class_mask_transposed)
    
    #print('mask', class_mask_transposed)
    
    #print('losses', losses.unsqueeze(0))
    
    #print(class_loss)
    
    class_loss = class_loss / class_total.unsqueeze(0)
        
    return class_loss.squeeze()


def per_class_counting(dataset):

    class_counts = defaultdict(int)
    
    if isinstance(dataset.targets, list):
        # If targets are in list format
        for label in dataset.targets:
            class_counts[label] += 1
    elif isinstance(dataset.targets, torch.Tensor):
        # If targets are in tensor format
        for label in dataset.targets.tolist():
            class_counts[label] += 1
    else:
        raise ValueError("Unsupported data type for targets. Must be a list or a torch.Tensor.")
    
    return class_counts



def count_classes_in_batches(dataloader):
    class_counts = defaultdict(int)
    for _, labels in dataloader:
        for label in labels:
            class_counts[label.item()] += 1
    return class_counts





class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        

    def forward(self, outputs, targets , reduction='mean'):
        losses = 1 - targets * outputs
        hinge_losses = torch.clamp(losses, min=0.0)  # max(0, 1 - y * o)
        
        if reduction == 'mean':
            return torch.mean(hinge_losses)
        elif reduction == 'sum':
            return torch.sum(hinge_losses)
        elif reduction == 'none':
            return hinge_losses
        else:
            raise ValueError("Invalid reduction mode. Use 'mean', 'sum', or 'none'.")






class ImageClassificationBase(nn.Module):
    def training_step(self, batch, num_data_points, params):
        images, labels = batch 
        #images = images.double()
        out = self(images)                  # Generate predictions (because ResNet9 (the model class inherit ImageClassificationBase)): when you use self() in pytorch it looks for forward method (defined in ResNet9 class)
        if params['Loss_function']=='Hinge':
            hinge = HingeLoss()
            loss = hinge(out.squeeze(), labels, reduction='mean')
        elif params['Loss_function']=='CrossEntropy':
            loss = F.cross_entropy(out, labels) # Calculate loss
        
        return {'loss': loss}#{'loss': loss, 'train_f': f}

    def StoringVariablesCreation(self):
        self.TrainLoss=[]
        self.ValLoss=[]
        self.TrainAcc=[]
        self.ValAcc=[]
        self.Trainf0=[]
        self.TrainMaxf=[]
        self.ValMaxf=[]
        self.Valf0=[]
        self.Trainfs=[]
        self.TrainOrderedfs=[]
        self.Valfs=[]
        self.ValOrderedfs=[]
        
        self.TrainPCLoss=[]
        self.ValPCLoss=[]
        self.TrainPCAcc=[]
        self.ValPCAcc=[]  
        
        self.GradNorm=[]
        self.GradNorm.append(0)
        self.StepSize=[]
        
        self.Weights_Layer_grad = []
        self.Biases_Layer_grad = []

        
        self.time=[]

    #defining same methods as below restricted to the guesses computation in order to repeat the computation for the train set
    """
    #included in training_step
    @torch.no_grad()
    def guesses_measure(self, batch, num_data_points, label_list): 
        images, labels = batch 
        out = self(images)  
        f = guesses(out, num_data_points, label_list)
    """
    def dataset_guess_wrap(self, outputs):
        """
        Combines the outputs of multiple batches and updates the training statistics.

        Parameters:
        - outputs: A list of tensors representing the outputs of each batch.

        Returns:
        None
        """
        batch_fs = [x for x in outputs]
        #print('fraz di batches', batch_fs)
        epoch_f = torch.sum(torch.stack(batch_fs), dim=0)      # Combine accuracies  
        self.Trainf0.append(epoch_f[0])
        self.TrainMaxf.append(np.max(epoch_f.numpy()))
        
           
    #@torch.no_grad() #not necessary to use this decorator as we cn substitute it with torch.set_grad_enabled() we have however to be careful in confine the change in the status only within the function's scope
    def validation_step(self, en_batch, num_data_points, mode, params):
        GradFlag={'KickStart': True, 'Train': False, 'Eval': False}
        
        @contextmanager
        def temporary_grad_enabled(flag):
            original_status = torch.is_grad_enabled()
            torch.set_grad_enabled(flag)
            yield
            torch.set_grad_enabled(original_status)

        
        with temporary_grad_enabled(GradFlag[mode]):
            
            if mode=='Train' or mode=='KickStart':
                self.train()
                classes_num = params['train_classes']
            elif mode=='Eval':
                self.eval()
                classes_num = params['valid_classes']
            num_classes = len(params['label_list'])
            #print("CHECK", label_list, num_classes)
            n_batch, batch = en_batch
            images, labels = batch 
            #images = images.double() 
            out = self(images)                    # Generate predictions
            
            if params['Loss_function']=='Hinge':
                hinge = HingeLoss()
                #print('dimension: ', out.squeeze(), labels, labels*out.squeeze())
                loss = hinge(out.squeeze(), labels, reduction='none')
            elif params['Loss_function']=='CrossEntropy':
                loss = F.cross_entropy(out, labels, reduction='none') # Calculate loss
            
            pc_loss = per_class_loss(loss, labels, num_classes, classes_num, params)
            
            if mode=='KickStart':
                #define a dummy temp optimizer just to reset gradients after grad norm storing 
                optimizer = opt_func(self.parameters(), 0.)

                BatchGradNorm=0
                loss.mean().backward()
                
                #for the first Batch we save the grad layer norm at init

                if n_batch==0:
                    LayerNormGrad(self)
                
                    
                
                parameters = [p for p in self.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    BatchGradNorm += param_norm.item() ** 2
                BatchGradNorm = BatchGradNorm ** 0.5  
                
                self.GradNorm[0] += BatchGradNorm/params['num_tr_batches']
                
                optimizer.zero_grad() #reset gradient before next step
                
            
            loss = torch.sum(loss) #reaggregate the loss for the global stats

            
            acc = accuracy(out, labels, params)          # Calculate accuracy
            pc_acc = per_class_acc(out, labels, num_classes, classes_num, params)
            f = guesses(out, num_data_points, params, n_batch) #Calculate class predictions according to the out values
        return {'val_loss': loss.detach(), 'val_acc': acc, 'val_f': f, 'pc_loss':pc_loss.detach(), 'pc_acc':pc_acc}
        
    def validation_epoch_end(self, outputs, mode):
        if mode=='Train':
            self.train()
        elif mode=='Eval':
            self.eval()
        
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        if mode=='Train':
            self.TrainLoss.append(epoch_loss.item())
        elif mode=='Eval':
            self.ValLoss.append(epoch_loss.item())
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        if mode=='Train':
            self.TrainAcc.append(epoch_acc.item())
        elif mode=='Eval':
            self.ValAcc.append(epoch_acc.item())
            
        batch_pc_accs = [x['pc_acc'] for x in outputs]
        epoch_pc_accs = torch.sum(torch.stack(batch_pc_accs), dim=0)
        if mode=='Train':
            self.TrainPCAcc.append((epoch_pc_accs.cpu()).numpy())
        elif mode=='Eval':    
            self.ValPCAcc.append((epoch_pc_accs.cpu()).numpy())
        
        batch_pc_losses = [x['pc_loss'] for x in outputs]
        epoch_pc_losses = torch.sum(torch.stack(batch_pc_losses), dim=0)
        if mode=='Train':
            self.TrainPCLoss.append((epoch_pc_losses.cpu()).numpy())
        elif mode=='Eval':    
            self.ValPCLoss.append((epoch_pc_losses.cpu()).numpy())
        
        
        
        batch_fs = [x['val_f'] for x in outputs]
        #print('fraz di batches', batch_fs)
        epoch_f = torch.sum(torch.stack(batch_fs), dim=0)      # Combine accuracies  
        if mode=='Train':
            self.Trainf0.append(epoch_f[0].item())
            self.TrainMaxf.append(np.max(epoch_f.numpy()))  
            self.Trainfs.append(epoch_f.numpy())
            self.TrainOrderedfs.append(np.sort(epoch_f.numpy())[::-1])

        elif mode=='Eval':
            self.Valf0.append(epoch_f[0].item())
            self.ValMaxf.append(np.max(epoch_f.numpy()))
            self.Valfs.append(epoch_f.numpy())
            self.ValOrderedfs.append(np.sort(epoch_f.numpy())[::-1])
        
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'val_f': epoch_f}
    









@torch.no_grad()
def evaluate(model, data_loader, mode, params):
    #model.eval() #the decoration above already take care of the gradient not saved; you can comment this to keep the batch norm active during the evaluation
    outputs = [model.validation_step(batch, params['num_data_points'][mode], mode, params) for batch in enumerate(data_loader)]
    return model.validation_epoch_end(outputs, mode)



def BatchGradNorm(net):
    """
    Normalize the gradient vector 

    Parameters
    ----------
    net : pytorch architecture
        model to retrive the parameter and the corresponding grad.

    Returns
    -------
    None.

    """
    
    Norm =0
    for p in net.parameters():
        param_norm = p.grad.detach().data.norm(2)
        Norm += param_norm.item() ** 2   
    for p in net.parameters():
        p.grad = torch.div(p.grad.clone(), Norm**0.5)
    



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def save_on_file(model, params):
    
    with open(params['FolderPath'] + "/Time.txt", "w") as f:
        np.savetxt(f, np.array(model.time), delimiter = ',') 
    
    with open(params['FolderPath'] + "/ValidLoss.txt", "w") as f:
        np.savetxt(f, np.array(model.ValLoss), delimiter = ',')

    with open(params['FolderPath'] + "/ValidAcc.txt", "w") as f:
        np.savetxt(f, np.array(model.ValAcc), delimiter = ',')         

    with open(params['FolderPath'] + "/Validf0.txt", "w") as f:
        np.savetxt(f, np.array(model.Valf0), delimiter = ',')  
        
    with open(params['FolderPath'] + "/ValidMaxf.txt", "w") as f:
        np.savetxt(f, np.array(model.ValMaxf), delimiter = ',')  

    with open(params['FolderPath'] + "/ValidClassesf.txt", "w") as f:
        np.savetxt(f, np.array(model.Valfs), delimiter = ',')  
        
    with open(params['FolderPath'] + "/ValidClassesOrderedf.txt", "w") as f:
        np.savetxt(f, np.array(model.ValOrderedfs), delimiter = ',')  

    with open(params['FolderPath'] + "/TrainLoss.txt", "w") as f:
        np.savetxt(f, np.array(model.TrainLoss), delimiter = ',')

    with open(params['FolderPath'] + "/TrainAcc.txt", "w") as f:
        np.savetxt(f, np.array(model.TrainAcc), delimiter = ',')   

    with open(params['FolderPath'] + "/Trainf0.txt", "w") as f:
        np.savetxt(f, np.array(model.Trainf0), delimiter = ',')  
        
    with open(params['FolderPath'] + "/TrainMaxf.txt", "w") as f:
        np.savetxt(f, np.array(model.TrainMaxf), delimiter = ',')  
        
    with open(params['FolderPath'] + "/TrainClassesf.txt", "w") as f:
        np.savetxt(f, np.array(model.Trainfs), delimiter = ',')  
        
    with open(params['FolderPath'] + "/TrainClassesOrderedf.txt", "w") as f:
        np.savetxt(f, np.array(model.TrainOrderedfs), delimiter = ',')  


    with open(params['FolderPath'] + "/TrainClassesLoss.txt", "w") as f:
        np.savetxt(f, np.array(model.TrainPCLoss), delimiter = ',')      
        
    with open(params['FolderPath'] + "/ValidClassesLoss.txt", "w") as f:
        np.savetxt(f, np.array(model.ValPCLoss), delimiter = ',')     

    with open(params['FolderPath'] + "/TrainClassesAcc.txt", "w") as f:
        np.savetxt(f, np.array(model.TrainPCAcc), delimiter = ',')      
        
    with open(params['FolderPath'] + "/ValidClassesAcc.txt", "w") as f:
        np.savetxt(f, np.array(model.ValPCAcc), delimiter = ',')    
        
        
    with open(params['FolderPath'] + "/GradNorm.txt", "w") as f:
        np.savetxt(f, np.array(model.GradNorm), delimiter = ',') 
        
    with open(params['FolderPath'] + "/Weights_Layer_grad.txt", "w") as f:
        np.savetxt(f, np.array(model.Weights_Layer_grad), delimiter = ',')   
        
    with open(params['FolderPath'] + "/Biases_Layer_grad.txt", "w") as f:
        np.savetxt(f, np.array(model.Biases_Layer_grad), delimiter = ',')
        
    with open(params['FolderPath'] + "/StepSize.txt", "w") as f:
        np.savetxt(f, np.array(model.StepSize), delimiter = ',') 
        






class GaussBlobsDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.samples = []
        self.class_map = {}
        self.targets = []
        self.transform = transform

        for family in os.listdir(data_root):
            family_folder = os.path.join(data_root, family)

            #create a mapping for the classes
            self.class_map[family] = int(family)

            for sample in os.listdir(family_folder):
                sample_filepath = os.path.join(family_folder, sample)
                self.samples.append((family, torch.load(sample_filepath)))
                
                self.targets.append(self.class_map[family]) 


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_name, data = self.samples[idx]
        #class_id = self.class_map[class_name] #temporary changed with the line below as we use the target attribute to access the labels (e.g. for label mapping)
        class_id = self.targets[idx]
        
        
        #we don't need to convert label in tensor here; dataloader will do it for us (creating a batch tensor label)
        #class_id = torch.tensor([class_id])
        
        #return self.samples[idx]
        

        if self.transform:
            data = self.transform(data)        
        
        return data, class_id

    def get_sample_shape(self, index):
        # Get the shape of a data sample at the given index
        sample = self[index][0]
        return sample.shape
    


class ShiftTensor(object):
    def __init__(self, shift_const):
        self.shift_const = shift_const

    def __call__(self, tensor):
        # Shift the tensor by the constant value
        shifted_tensor = tensor + self.shift_const
        return shifted_tensor







@torch.no_grad()   #we just need output stas here, not gradient; so we add this decoration to avoid its computation
def InitVarComp(N_in, Stats_List, train_loader, par, N_Init, model):
    """
    

    Parameters
    ----------
    N_in : number of initializations used to collect the stats
        DESCRIPTION.
    Stats_List : an empty list that will be filled with the collected stats
        DESCRIPTION.

    Returns
    -------
    None.

    """   
    Means = []
    Stds = []
    res = {}
    for i in range (0,N_Init):
        Stats_List.append([]) #create a list to store stats for the cycle
        if par['Architecture']=='ViT':
            model.apply(model.initialize_weights)
        else:
            model.initialize_weights() #re-initialize the network with new set of weights
        
        for batch in train_loader:
            images, labels = batch 
            Stats_List[i].append(model(images)[:, 0]) #appending the oututs corresponding to one of the nodes (the 0)
            
        Stats_List[i] = torch.cat(Stats_List[i], dim=0)
        Means.append(torch.mean(Stats_List[i]).item())
        Stds.append(torch.std(Stats_List[i]).item())
    #print(Stats_List)
    mean_of_stds = np.mean(Stds)
    std_of_means = np.std(Means)
    res['dataset_output_std'] = mean_of_stds
    res['ensemble_centers_std'] = std_of_means

    print(res)
    return res




@torch.no_grad()   #we just need output stas here, not gradient; so we add this decoration to avoid its computation
def SingleInitStats(train_loader, par, model):
    """
    Compute statistics for a single network initialization.
    
    Parameters
    ----------
    train_loader : DataLoader
        The data loader for the training set.
    par : dict
        Dictionary of parameters including the network architecture.

    Returns
    -------
    res : dict
        Dictionary containing the computed statistics:
            - 'output_centers_distance': The mean distance between the centers of the output nodes.
            - 'mean_output_variance': The mean of the variances of the output nodes.
    """

    # Initialize the network
    if par['Architecture'] == 'ViT':
        model.apply(model.initialize_weights)
    else:
        model.initialize_weights()

    num_batches = 0
    output_means_sum = None
    output_vars_sum = None
    total_samples = 0


    for batch in train_loader:
        images, labels = batch
        outputs = model(images)
        batch_size = outputs.size(0)
        
        if output_means_sum is None:
            # Initialize the accumulators with the correct shape
            output_means_sum = torch.zeros(outputs.size(1)).to(outputs.device)
            output_vars_sum = torch.zeros(outputs.size(1)).to(outputs.device)
        
        # Accumulate means and variances for each output node
        output_means_sum += torch.sum(outputs, dim=0)
        output_vars_sum += torch.sum(outputs ** 2, dim=0)
        total_samples += batch_size
        num_batches += 1

    # Compute the overall means and variances
    output_means = output_means_sum / total_samples
    output_vars = (output_vars_sum / total_samples) - (output_means ** 2)

    # Compute the mean distance between the centers of the output nodes
    num_output_nodes = output_means.size(0)
    if num_output_nodes == 2:
        output_centers_distance = torch.abs(output_means[0] - output_means[1]).item()
    else:
        # For more than two output nodes, compute the mean pairwise distance between centers
        distances = []
        for i in range(num_output_nodes):
            for j in range(i + 1, num_output_nodes):
                distances.append(torch.abs(output_means[i] - output_means[j]).item())
        output_centers_distance = np.mean(distances)
    
    # Compute the mean of the output variances
    mean_output_variance = torch.mean(output_vars).item()

    print('Output means: ', output_means, 'their distance: ', output_centers_distance)
    print('Output vars: ', output_vars, 'their mean value: ', mean_output_variance)

    print('THE RATIO:', output_centers_distance/mean_output_variance)

    # Store the results in a dictionary
    res = {
        'output_centers_distance': output_centers_distance,
        'mean_output_variance': mean_output_variance
    }
    
    print(res)
    return res





def label_map_function(fi, params):
    """
    generate a label map based on the value of the IGB; in particular we order the components according to a descending order of IGB

    Parameters
    ----------
    fi : torch tensor
        tensor of fractions fi computed on the training set just after initializzation.

    Returns
    -------
    result_dict : dict
        label map to order the labels according to fi.

    """
    print("Received params:", params)
    print("params['label_list']:", params['label_list'])

    sorted_indices = torch.argsort(fi, descending=True)
    #print('SORTED INDECES', sorted_indices)
    print("params['label_list']:", params['label_list'])
    print("Type of params['label_list']:", type(params['label_list']))
    for item in params['label_list']:
        print(f"Item: {item}, Type: {type(item)}")
    # Print the type of sorted_indices
    print("Type of sorted_indices:", type(sorted_indices))

    # Print the content and type of each element in sorted_indices
    for index in sorted_indices:
        print(f"Index: {index}, Type: {type(index)}, Index.item(): {index.item()}, Type of index.item(): {type(index.item())}")

    result_dict = {params['label_list'][index.item()]: params['label_list'][i] for i, index in enumerate(sorted_indices)}
    
    return result_dict

def save_image_and_label(data_point, image_path, label_path):
    # Saving the image and label information into a text file
    image, label=data_point
    with open(label_path, 'w') as file:
        file.write(f'Label: {label}\n')

    TF.to_pil_image(image).save(image_path)



def OrderOutputNodes(model, params):

    label_map = params['label_map']
    #print('label_map', label_map, params['label_map'][1])
    
    if params['Loss_function']=='Hinge':

        #CHANGE LAST FULLY CONNECTION ACCORDING TO IGB IN ORDER TO ORDER NODES IN DECREASING ORDER OF IGB
        
        # Extract the weights from the original linear layer
        original_weights = model.output.weight.data   
        

        
        # Initialize a new weight matrix to reconstruct based on the swap dictionary
        new_weights = original_weights*params['label_map'][1] #if the class has to be inverted we change sign to the last layer

        #print('weights', original_weights, new_weights, params['label_map'][1])

        # Ensure the new weights have the same data type as the original weights
        new_weights = new_weights.to(dtype=original_weights.dtype)
        
        
        # Create a new linear layer and replace the weights with the newly constructed matrix
        new_output_layer = nn.Linear(model.prev_size, model.n_outputs).to(original_weights.device)
        new_output_layer.weight.data = new_weights.to(original_weights.dtype)

        # Replace the original output layer in the model with the new one
        model.output = new_output_layer
    else:
    
        #CHANGE LAST FULLY CONNECTION ACCORDING TO IGB IN ORDER TO ORDER NODES IN DECREASING ORDER OF IGB
        
        # Extract the weights from the original linear layer
        original_weights = model.output.weight.data
        
        # Initialize a new weight matrix to reconstruct based on the swap dictionary
        new_weights = torch.zeros_like(original_weights).to(original_weights.device)
            
        # Iterate through the dictionary and build the new weight matrix
        for new_row_idx, row_idx in label_map.items():
            new_weights[new_row_idx] = original_weights[row_idx]
        
        # Ensure the new weights have the same data type as the original weights
        new_weights = new_weights.to(dtype=original_weights.dtype)
        
        
        # Create a new linear layer and replace the weights with the newly constructed matrix
        new_output_layer = nn.Linear(model.prev_size, model.n_outputs).to(original_weights.device)
        new_output_layer.weight.data = new_weights.to(original_weights.dtype)
            
        # Replace the original output layer in the model with the new one
        model.output = new_output_layer


def convert_to_tensor_if_needed(targets):
    if isinstance(targets, list):
        return torch.tensor(targets)
    elif isinstance(targets, torch.Tensor):
        return targets  # If it's already a tensor, no conversion needed
    else:
        raise ValueError("Unsupported data type for targets. Must be a list or a torch.Tensor.")



#TODO: complete the following function to save grads
def LayerNormGrad(model):

    # Compute the gradient norms for each layer (weights and biases)
    weight_gradient_norms = [] #first we re-empty the list that we will fill with the gradients of the batch
    bias_gradient_norms = [] #first we re-empty the list that we will fill with the gradients of the batch

    for name, param in model.named_parameters():
        if 'weight' in name:
            #print(name, param)
            #gradient_norm = param.grad.norm().item()
            gradient_norm = torch.mean(torch.abs(param.grad)).item()# we try with the average of absolute values of the single components

            #print(name, ' norm: ', gradient_norm)
            weight_gradient_norms.append(gradient_norm)
        
        elif 'bias' in name:
            gradient_norm = param.grad.norm().item()
            bias_gradient_norms.append(gradient_norm)
            


    # Stack the tensors into a single tensor sending them to cpu and save it as numpy array
        
    #print('weights layer norms', weight_gradient_norms)
    #print('biases layer norms', bias_gradient_norms)
    model.Weights_Layer_grad.append(np.array(weight_gradient_norms))
    model.Biases_Layer_grad.append(np.array(bias_gradient_norms))
    #once we have it saved we can store it 
    




#%% SET MODE FLAGS FOR MODEL FLEXIBILITY

#transform the input value token from outside rin a variable
p = argparse.ArgumentParser(description = 'Sample index')
p.add_argument('SampleIndex', help = 'Sample index')
p.add_argument('FolderName', type = str, help = 'Name of the main folder where to put the samples folders')
p.add_argument('learning_rate', type = float, help = 'learning rate (hyperparameter) selected for the run')
p.add_argument('batch_size', type = int, help = 'Batch size (hyperparameter) selected for the run')
p.add_argument('ks', type = int, help = 'kernel size for the MaxPool layer (if present)')
p.add_argument('Relu_Slope', type = float, help = 'slope to assign to the customized ReLU (if present)')
p.add_argument('Data_Shift_Constant', type = float, help ='shifting constant used for data standardization')



args = p.parse_args()

print('first parameter (run index) passed from the script is: ', args.SampleIndex)
print('second parameter (Output Folder) passed from the script is: ', args.FolderName)

print("The PID of the main code is: ", os.getpid())

ReLU_Slope= args.Relu_Slope
shift_const=args.Data_Shift_Constant

TrainMode = 'ON' #if 'ON' pefrorm also training otherwise just init stats

IGB_Mode='Off' #can be 'On' or 'Off' changing the activation function or elements used in the architecture

IGB_Sel_Mode='Off' #can be: 'Low', 'High' or 'Off'

IGB_Sel_criterion='GammaProxy' #Can be 'GammaProxy' (ratio between centers distances and variance) or 'Fractions' (comparison between fractions of assigned data-points)

OptimFlag = 'Adam' #can be: 'SGD' or 'Adam'

FixedStepSizeFlag= 'OFF' # can be 'ON' or 'OFF' depending if we want or not to the value FixStepSize

FixStepSize= 0.0005

Architecture = 'MLP_mixer2' #can be 'MLP' or 'MLP-mixer' or 'BaseCNN' or 'ViT' or 'ResNet' 'MLP_mixer2'

Loss_function='CrossEntropy' #can be hinge or CrossEntropy or Hinge

ds = 'CatsVsDogs' #options so far: 'Gaussian', 'CatsVsDogs', 'ImbalancedGaussian' or 'Cifar10'

NormMode = 'BN' #4 possible modes: 'Shift'==add an offset to center the activated nodes, 'OFF'==No normalization, 'InitShift' == shift computed on the first batch and same statistics used for all the train steps 'BN' classic batch norm

NormPos = 'Before' # determine the position of norm. layer (w.r.t. the activation function). 2 possible modes: 'After' or 'Before'

OrderingClassesFlag='ON' #when set 'ON' order the weights raws of the last layer according to the IGB values such that classes with bigger guessing fraction have smaller node index

GradNormMode='OFF' #if 'ON' normalize the grads 

SplittingStats='OFF' #if 'ON' splits the stats according to the class with biggest IGB value

AugmentationFlag='ON'


SigmasStatsFlag='OFF' #set a flag to decide if to compute prel. Stat to estimate the level of IGB (by gamma) on the ensemble: it can be 'ON' or 'OFF'



ImbalanceFactor= 1. #if we want to select a subset of elements for each class according to an imbalance factor (set on 1 to have balanced datasets)

epochs = 20000
learning_rate = args.learning_rate
grad_clip = None #0.1
weight_decay = 0. #1e-4

if OptimFlag=='SGD':
    opt_func = torch.optim.SGD
elif OptimFlag=='Adam':
    opt_func = torch.optim.Adam
    
    
batch_size = args.batch_size
ks = args.ks

#opt_func = torch.optim.AdamW
#grad_clip = 0.1
#weight_decay = 1e-4
#opt_func = torch.optim.Adam





L=10
hidden_sizes = [100]*L #[3000]*L  # Replace with your desired hidden layer sizes

n_outputs = 2  # Replace with the number of classes in your custom dataset

if Loss_function=='Hinge':
    n_outputs=1











#%%% CIFAR10 augmentation  

#from https://github.com/omihub777/ViT-CIFAR/blob/main/autoaugment.py

class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img











