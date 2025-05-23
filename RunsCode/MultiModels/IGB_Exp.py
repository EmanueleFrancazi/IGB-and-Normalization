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
import os

# Dynamically add the project root (two levels up) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)


import time

import copy


#for pre-trained architectures
import torchvision.models as models
import torchvision.transforms as transforms


from utils.IGB_utils import *


#fixing initial times to calculate the total time of the cycle
start_TotTime = time.time()






#%% Setting Random seed

def FixSeed(seed):
    """
    initialize the seed for random generator used over the run : we have to do it for all the libraries that use on random generator (yorch, random and numpy)

    Parameters
    ----------
    seed : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    torch.manual_seed(seed) #fixing the seed of 'torch' module
    random.seed(seed) #fixing the seed of 'random' module
    np.random.seed(seed) #fixing the seed of 'numpy' module   

#HYPERPARAMETERS
CheckMode = 'OFF' #this flag active ('ON') or deactive ('OFF') the checking mode (used for debugging purposes)
#NOTE:Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. 
#Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.
#However, there are some steps you can take to limit the number of sources of nondeterministic behavior for a specific platform, device, and PyTorch release.
if CheckMode=='ON':#when we are in the checking mode we want to reproduce the same simulation to check the new modified code reproduce the same behaviour
    seed = 0
    FixSeed(seed)
elif CheckMode=='OFF':
    #creation of seeds for usual simulations
    #WARNING: use the time of the machine as seed you have to be sure that also for short interval between successive interval you get different seeds
    #with the following choice  for very short periods of time, the initial seeds for feeding the pseudo-random generator will be hugely different between two successive calls
    t = int( time.time() * 1000.0 )
    seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >>  8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)   
    
    #if the above syntax should be confusing:
    """
    Here is a hex value, 0x12345678, written as binary, and annotated with some bit positions:
    
    |31           24|23           16|15            8|7         bit 0|
    +---------------+---------------+---------------+---------------+
    |0 0 0 1 0 0 1 0|0 0 1 1 0 1 0 0|0 1 0 1 0 1 1 0|0 1 1 1 1 0 0 0|
    +---------------+---------------+---------------+---------------+
    
    ...and here is 0x000000FF:
    
    +---------------+---------------+---------------+---------------+
    |0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|1 1 1 1 1 1 1 1|
    +---------------+---------------+---------------+---------------+
    
    So a bitwise AND selects just the bottom 8 bits of the original value:
    
    +---------------+---------------+---------------+---------------+
    |0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|0 1 1 1 1 0 0 0|
    +---------------+---------------+---------------+---------------+
    
    ...and shifting it left by 24 bits moves it from the bottom 8 bits to the top:
    
    +---------------+---------------+---------------+---------------+
    |0 1 1 1 1 0 0 0|0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|
    +---------------+---------------+---------------+---------------+
    
    ...which is 0x78000000 in hex.
    
    The other parts work on the remaining 8-bit portions of the input:
    
      0x12345678
    & 0x000000FF
      ----------
      0x00000078 << 24 = 0x78000000       (as shown above)
    
      0x12345678
    & 0x0000FF00
      ----------
      0x00005600 <<  8 = 0x00560000
    
      0x12345678
    & 0x00FF0000
      ----------
      0x00340000 >>  8 = 0x00003400
    
      0x12345678
    & 0x00000000
      ----------
      0x12000000 >> 24 = 0x00000012
    
                       | ----------
                         0x78563412
    
    so the overall effect is to consider the 32-bit value ldata as a sequence of four 8-bit bytes, and reverse their order.
        
    """
    
    
    
    FixSeed(seed)
    



#%% USEFUL FUNCTIONS


def fit_one_cycle(epochs, ValChecks,max_lr, model, train_loader, val_loader, params,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    
    with torch.cuda.device(params['device']): # torch.cuda.empty_cache() write data to gpu0 (by default): you can get a memory error if gpu:0 is fully occupied. with this line no memory allocation occurs on gpu0.
        torch.cuda.empty_cache()
    
    history = []
    
    # Set up cutom optimizer with weight decay
    
    #optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)   # Set up cutom optimizer with weight decay
    
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader)) # Set up one-cycle learning rate scheduler
    
    if params['Architecture']=='MLP_mixer2' or params['Architecture']=='ViT':
        optimizer = opt_func(model.parameters(), lr=max_lr, betas=(0.9, 0.99), weight_decay=5e-5)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'], eta_min=1e-6)
        sched = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler=base_scheduler)
    # Set up one-cycle learning rate scheduler
    #sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    step=0
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        optimizer.zero_grad()
        train_losses = []
        #lrs = []
        TrRes=[]
        for batch in train_loader:
            Res= model.training_step(batch, num_trdata_points, params)
            loss = Res['loss']
            #TrRes.append(Res['train_f'])
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            if params['GradNormMode']=='ON':
                BatchGradNorm(model)
            
            optimizer.step() #update weights
            
            


            if (step+1) in ValChecks:      # Validation phase

                model.LR.append(sched.get_last_lr()[0]) #save learning rate

                
                #first we save the norm of the gradient used for the step and the corresponding step size
                total_norm = 0
                parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5  
                
                model.GradNorm.append(total_norm)
                #we then multiply the gradnorm by the learning rate to get the step size
                model.StepSize.append(total_norm*optimizer.param_groups[-1]['lr']) #this works because in our caase all params group have the same learning rate
                
                #test eval
                test_result = evaluate(model, val_loader, 'Eval', params)
                test_result['train_loss'] = torch.stack(train_losses).mean().item()
                model.time.append(step+1)

                
                #train eval
                train_result = evaluate(model, train_loader, 'Train', params)
                WandB_logs(step+1, model) #log on wandb 
                save_on_file(model, params) #save stats on files
                
                
                history.append(test_result) 

            
            step+=1
            optimizer.zero_grad() #reset gradient before next step
        #model.dataset_guess_wrap(TrRes)
        
        # Record & update learning rate
        #lrs.append(get_lr(optimizer))
        
        sched.step()

        # Log the current learning rate and T_cur
        current_lr = sched.get_last_lr()[0]  # Get the current learning rate
        print(f"Epoch {epoch}: Learning Rate = {current_lr}")

        
    #print('last step performed was: ', step)
        

    return history






#%% PREPARING DATA




"""
# Dowload the dataset
dataset_url = "http://files.fast.ai/data/examples/cifar10.tgz"
download_url(dataset_url, '.')

# Extract from archive
with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    #tar.extractall(path='./data')
    tar.extractall(path='/home/EAWAG/francaem/restored/data/Cifar10_Kaggle_link')
"""    


# Look into the data directory
if ds == 'CatsVsDogs':
    #data_dir = '/home/EAWAG/francaem/restored/Prova/IGB/ICLR_rebuttal/Kaggle/data/Cat_vs_Dog/cifar10'
    data_dir='/home/EAWAG/francaem/restored/data/Cifar10_Kaggle_link/Cats_Dogs'
    #data_dir = '/cluster/home/efrancazi/Data/Cat_vs_Dog/cifar10'
elif ds == 'Cifar10':
    data_dir = '/home/EAWAG/francaem/restored/data/Cifar10_Kaggle_link/cifar10'
elif ds == 'Gaussian':
    data_dir = "/home/EAWAG/francaem/restored/data/GaussianBlobs/"
    
elif ds == 'ImbalancedGaussian':
    data_dir = "/home/EAWAG/francaem/restored/data/ImbalancedGaussianBlobs/"



#%% DATA PRE-PROCESSING 




# Data transforms (normalization & data augmentation)
stats = ((0.4914 + shift_const, 0.4822 + shift_const, 0.4465 + shift_const), (0.2023, 0.1994, 0.2010))

train_transform = []

train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])



#TODO: TIDY THIS AUGMENTATION PROCEDURE
if AugmentationFlag=='ON':

    
    train_transform += [tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                                 tt.RandomHorizontalFlip()]
    
    train_transform.append(CIFAR10Policy())
    
else:
    train_transform += [ tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                             tt.RandomHorizontalFlip()]
    


train_transform += [tt.ToTensor(), tt.Normalize(*stats,inplace=True)]    

train_tfms = tt.Compose(train_transform)

valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])


custom_transform = ShiftTensor(shift_const)
G_transforms = tt.Compose([custom_transform])

#G_stats = [shift_const,shift_const, shift_const]
#G_transforms = tt.Compose([tt.ToTensor(),tt.Normalize(mean=G_stats, std=[1.,1.,1.])])


# PyTorch datasets
#ImageFolder takes each folder, assigns the same label to all images in that folder. 
if (ds == 'Gaussian' or ds == 'ImbalancedGaussian'):
    train_ds = GaussBlobsDataset(data_dir, G_transforms)
    valid_ds = GaussBlobsDataset(data_dir, G_transforms) #using same data for train and valid just for debug; in principle should use different data
else:
    train_ds = ImageFolder(data_dir+'/train', train_tfms)
    valid_ds = ImageFolder(data_dir+'/test', valid_tfms)
    
    
#print(train_ds[0][0].type())


train_ds.targets = convert_to_tensor_if_needed(train_ds.targets)
valid_ds.targets = convert_to_tensor_if_needed(valid_ds.targets)

#create deep copy of targets attribute (useful for the label mapping phase)
traintargets = copy.deepcopy(train_ds.targets) 
validtargets = copy.deepcopy(valid_ds.targets) 


#define the number of datapoints in the dataset (if you look at len(dataloader) instead you get the number of batches)
num_trdata_points = len(train_ds)
num_valdata_points = len(valid_ds)

print('number of data points. Train : ', num_trdata_points, 'Valid : ', num_valdata_points)

train_classes = per_class_counting(train_ds)

valid_classes = per_class_counting(valid_ds)

if Loss_function=='Hinge':
    label_list=[-1, 1] #for the hinge we have to map the labels
else:
    label_list = list(train_classes.keys())
print('the classes for train and valid are: ', train_classes, valid_classes, label_list)

#print('targets before', dict(Counter(train_ds.targets.tolist())))


num_classes = len(train_classes)

#define now an array of imbalance ratio for the multiclass case
ImabalnceProportions = np.zeros(num_classes)
for i in range (0,num_classes):
    ImabalnceProportions[i] = ImbalanceFactor**i
    #modify the number of elements selected for each class
    
    #TODO: the current code assume a list of classes ordered in the range 0, num_classes-1 to ensure this you can perform a preliminary mapping to map each class (key of train_classes) in the range 0, num_classes-1
    train_classes[i] = train_classes[i]*ImabalnceProportions[i]

print('the classes for train and valid are (according to the desired imbalance ratio): ', train_classes, valid_classes, label_list)

#we now define a subset of indeces that will be used for our dataloader (for training set)

TrainIdx=None
for i in range (0,num_classes):
    trainTarget_idx = (traintargets==i).nonzero() 
    if TrainIdx==None:
        TrainIdx = trainTarget_idx[:][0:int(train_classes[i])]
    else:
        TrainIdx = torch.cat((TrainIdx, trainTarget_idx[:][0:int(train_classes[i])]),0)

#define the sampler for our dataset
train_sampler = SubsetRandomSampler(TrainIdx) 



if Loss_function=='Hinge':
    label_map ={-1:0, 1:1}
    for key in label_map:
        #print(key, label_map[key], train_ds.targets[0], dict(Counter(train_ds.targets.tolist())))
        train_ds.targets[train_ds.targets==label_map[key]]= key
        valid_ds.targets[valid_ds.targets==label_map[key]]= key
        
    #print('targets after', dict(Counter(train_ds.targets.tolist())))


#set the imbalance on the train ds selecting a subset of elements according to the chosen ImbalanceRatio

    


# PyTorch data loaders
if (ds=='Gaussian'or ds == 'ImbalancedGaussian'):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size*2) 
else:
    #train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)   #use this to check if you get the same observable in case of no label swaps (as sanity check)
    #train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    #valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)
    train_dl = DataLoader(train_ds, batch_size, sampler = train_sampler, num_workers=0, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3,pin_memory=True)


#we verify that we are correctly using images from the sampler subset 

Load_checkPoint = time.time()

Load_Time_diff = Load_checkPoint - start_TotTime

print('TIME till data loading is: ', Load_Time_diff)

actual_class_distribution = count_classes_in_batches(train_dl)

print("Actual Class Distribution in Batches:", actual_class_distribution)


num_trdata_points = len(train_dl.sampler)

print('number of data points after sampler', num_trdata_points)


sample_shape = train_ds[0][0].shape

input_size = np.prod(sample_shape)
print(input_size)



#fix the checkpoints for validation based on the total number of steps
num_tr_batches = len(train_dl)
N_ValidSteps=30
TimeValSteps= ValidTimes(epochs, num_tr_batches, N_ValidSteps)
print('epochs with validation: ', TimeValSteps)







#%%Collecting Variables (some of them come from utils.IGB_utils)


params = {'NormMode': NormMode,  'hidden_sizes': hidden_sizes, 'n_outputs': n_outputs, 'input_size': input_size, 'NormPos': NormPos
          , 'Architecture': Architecture
          ,'ks':ks, 'ReLU_Slope': ReLU_Slope, 'Loss_function': Loss_function, 'IGB_Mode':IGB_Mode
          , 'train_classes': train_classes, 'valid_classes':valid_classes, 'num_data_points': {'Train': num_trdata_points, 'Eval':num_valdata_points}, 'label_list':label_list
          ,'epochs':epochs, 'num_tr_batches':num_tr_batches
          , 'GradNormMode': GradNormMode,
          
          }









#%%SET GPU USE



def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
   
#device = get_default_device()
device = "cuda:0" if torch.cuda.is_available() else "cpu" 
#device = "cpu"
print(device)

params['device'] = device


#We can now wrap our training and validation data loaders using DeviceDataLoader for automatically transferring batches of data to the GPU (if available).


train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)








#%% Architecture


#%%% MLP

class CustomBatchNorm2d(nn.Module):
    def forward(self, x):
        """
        if self.training: #as for the BN we activate it only during training
            mean = x.mean(dim=0, keepdim=True) # Calculate mean along the batch dimension (dimension 0)
            return x - mean  # Subtract the mean from the input tensor
        else:
            return x
        """
        
    
        if self.training:
            mean = x.mean(dim=0, keepdim=True) # Calculate mean along the batch dimension (dimension 0)
            return x - mean  # Subtract the mean from the input tensor
        else:
            return x

        
class CustomInitBatchNorm2d(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.stored_means = {}  # Dictionary to store means for each layer
    
    def forward(self, x):
        if self.training and self.layer_id not in self.stored_means:
            self.stored_means[self.layer_id] = x.mean(dim=0, keepdim=True).detach()
        
        if self.layer_id in self.stored_means:
            return x - self.stored_means[self.layer_id]
        else:
            return x
    """
    def backward(self, grad_output):
        return grad_output  # Propagate gradients unchanged
    """




# define a customized ReLU to accentuate slope and get a bigger average value

def ScaledReLU(x, slope):
    m = nn.ReLU()
    return slope*m(x) #

class ScaReLU(nn.Module):

    def __init__(self, slope):
        '''
        Init method.
        '''
        super().__init__() # init the base class
        self.slope = slope

    

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return ScaledReLU(input, self.slope) # simply apply already implemented SiLU








# Define the architecture of the multilayer perceptron
class SimpleMLP(ImageClassificationBase):
    def __init__(self, params):
        super(SimpleMLP, self).__init__()
        layers = []
        input_size = params['input_size']
        hidden_sizes = params['hidden_sizes']
        self.n_outputs = params['n_outputs']
        NormFlag = params['NormMode']
        self.prev_size = input_size
        self.input_dim = input_size #save input as clss attribute so that we can use it for the forward
        #ks=1
        self.ScalReLU = ScaReLU(params['ReLU_Slope'])
        

        
        layer_counter=0
        for size in hidden_sizes:
            layers.append(nn.Linear(self.prev_size, size))
            #layers.append(nn.Tanh())   #to reproduce result deep information propagation

            if params['IGB_Mode']=='On':
                layers.append(nn.ReLU())
            elif params['IGB_Mode']=='Off':
                layers.append(nn.Tanh())
            #layers.append(self.ScalReLU)
            
            layers.append(nn.MaxPool1d(kernel_size=params['ks']))
            self.prev_size = math.ceil(size/params['ks'])
            
            #prev_size = size

            
            if NormFlag=='Shift':
                layers.append(CustomBatchNorm2d())
            elif NormFlag=='InitShift':
                layers.append(CustomInitBatchNorm2d(layer_id=layer_counter))
            
            layer_counter+=1
            
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(self.prev_size, self.n_outputs)

        self.initialize_weights() #calling the function below to initialize weights
        #self.weights_init() #call the orthogonal initial condition    
    def initialize_weights(self):
        #modules is the structure in which pytorch saves all the layers that make up the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        



    def forward(self, x):
        x = x.view(-1,self.input_dim)
        x = self.hidden(x)
        #print('before fc', x)
        x = self.output(x)
        #print('output', x)
        return x



#%%% MLP-mixer

#architecture source: https://github.com/jaketae/mlp-mixer

class MLP_MLPmix(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP_MLPmix(num_patches, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP_MLPmix(num_features, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.token_mixer = TokenMixer(
            num_patches, num_features, expansion_factor, dropout
        )
        self.channel_mixer = ChannelMixer(
            num_patches, num_features, expansion_factor, dropout
        )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, num_features)
        return x


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class MLPMixer(ImageClassificationBase):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        num_features=128,
        expansion_factor=2,
        num_layers=8,
        num_classes=10,
        dropout=0.5,
    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        # per-patch fully-connected is equivalent to strided conv2d
        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )
        self.mixers = nn.Sequential(
            *[
                MixerLayer(num_patches, num_features, expansion_factor, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(num_features, num_classes)
        
        self.prev_size=num_features
        self.n_outputs= num_classes


        self.initialize_weights() #calling the function below to initialize weights
        #self.weights_init() #call the orthogonal initial condition    
    def initialize_weights(self):
        #modules is the structure in which pytorch saves all the layers that make up the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)



    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mixers(patches)
        # embedding.shape == (batch_size, num_patches, num_features)
        embedding = embedding.mean(dim=1)
        logits = self.output(embedding)
        return logits



#%%% MLP-mixer v2

#from https://github.com/omihub777/MLP-Mixer-CIFAR/blob/main/README.md

class MLPMixer_v2(ImageClassificationBase):
    def __init__(self,params, in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, num_classes=10, drop_p=0., off_act=False, is_cls_token=False):
        super(MLPMixer_v2, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token

        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1


        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer_v2(num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act) 
            for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_size)

        self.output = nn.Linear(hidden_size, num_classes)

        self.prev_size=hidden_size
        self.n_outputs= num_classes

        self.initialize_weights() #calling the function below to initialize weights
        #self.weights_init() #call the orthogonal initial condition    
    def initialize_weights(self):
        #modules is the structure in which pytorch saves all the layers that make up the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)



    def forward(self, x):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        out = self.mixer_layers(out)
        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.output(out)
        return out


class MixerLayer_v2(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act):
        super(MixerLayer_v2, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)
    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        if params['NormPos']=='Before':
            out = self.do1(self.act(self.fc1(self.ln(x))))  
            out = self.do2(self.fc2(out))
        
        elif params['NormPos']=='After':
            out = self.do1(self.act(self.fc1(x)))
            out = self.do2(self.fc2(out))
            out = self.ln(out)
        else:
            print('NormPos Flag filled incorrectly; for this architecture only Before and After are possible states')
        
        
        return out+x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        if params['NormPos']=='Before':
            out = self.do1(self.act(self.fc1(self.ln(x))))
            out = self.do2(self.fc2(out))
        elif params['NormPos']=='After':
            out = self.do1(self.act(self.fc1(x)))
            out = self.do2(self.fc2(out))
            out = self.ln(out)
        else:
            print('NormPos Flag filled incorrectly; for this architecture only Before and After are possible states')
        
        return out+x

















#%%% Simple CNN

#source code: https://www.kaggle.com/code/grayphantom/cnn-on-cifar10-using-pytorch

class BaseCNNModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(), 
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(128*8*8,256),
            nn.ReLU(),
            nn.Linear(256,84),
            nn.ReLU()
            )
        self.prev_size=84
        self.n_outputs=2
        self.output=nn.Linear(self.prev_size,self.n_outputs)

        self.initialize_weights() #calling the function below to initialize weights
        #self.weights_init() #call the orthogonal initial condition    
    def initialize_weights(self):
        #modules is the structure in which pytorch saves all the layers that make up the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        x = self.network(x)
        x = self.output(x)
        return x



#%%%ResNet


#ResNET from https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min

class CustomBatchNorm2d(nn.Module):
    def forward(self, x):
        """
        if self.training: #as for the BN we activate it only during training
            mean = x.mean(dim=0, keepdim=True) # Calculate mean along the batch dimension (dimension 0)
            return x - mean  # Subtract the mean from the input tensor
        else:
            return x
        """
        mean = x.mean(dim=0, keepdim=True) # Calculate mean along the batch dimension (dimension 0)
        return x - mean  # Subtract the mean from the input tensor
    
class CustomInitBatchNorm2d(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.stored_means = {}  # Dictionary to store means for each layer
    
    def forward(self, x):
        if self.training and self.layer_id not in self.stored_means:
            self.stored_means[self.layer_id] = x.mean(dim=0, keepdim=True)
        
        if self.layer_id in self.stored_means:
            return x - self.stored_means[self.layer_id]
        else:
            return x

    def backward(self, grad_output):
        return grad_output  # Propagate gradients unchanged


def conv_block(in_channels, out_channels, NormFlag, layer_id, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    if NormPos=='Before':
        if NormFlag=='BN':
            layers.append(nn.BatchNorm2d(out_channels))
        elif NormFlag=='Shift':
            layers.append(CustomBatchNorm2d())
        elif NormFlag=='InitShift':
            layers.append(CustomInitBatchNorm2d(layer_id=layer_id))
            
    layers.append(nn.ReLU(inplace=True))




    if pool: layers.append(nn.MaxPool2d(2))

    if NormPos=='After':
        if NormFlag=='BN':
            layers.append(nn.BatchNorm2d(out_channels))
        elif NormFlag=='Shift':
            layers.append(CustomBatchNorm2d())
        elif NormFlag=='InitShift':
            layers.append(CustomInitBatchNorm2d(layer_id=layer_id))    

    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes, params):
        super().__init__()
        self.NormFlag = params['NormMode']
        
        self.conv1 = conv_block(in_channels, 64, self.NormFlag, 'conv1')
        self.conv2 = conv_block(64, 128, self.NormFlag, 'conv2', pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128, self.NormFlag, 'res1_1'), conv_block(128, 128, self.NormFlag, 'res1_2'))
        
        self.conv3 = conv_block(128, 256, self.NormFlag, 'conv3', pool=True)
        self.conv4 = conv_block(256, 512, self.NormFlag, 'conv4', pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512, self.NormFlag, 'res2_1'), conv_block(512, 512, self.NormFlag, 'res2_2'))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten())
        
        self.prev_size=512
        self.n_outputs=num_classes

        self.output= nn.Linear(self.prev_size, self.n_outputs)
        
        self.initialize_weights() #calling the function below to initialize weights
        #self.weights_init() #call the orthogonal initial condition    
    def initialize_weights(self):
        #modules is the structure in which pytorch saves all the layers that make up the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        out = self.output(out)
        return out

#%%% PRE-TRAINED ResNet50



class CustomResNet50(ImageClassificationBase):
    def __init__(self, num_classes=2, pretrained=True):
        super(CustomResNet50, self).__init__()
        # Load a pre-trained ResNet50 model
        if pretrained:
            self.pretrained_model = models.resnet50(pretrained=True)
        else:
            self.pretrained_model = models.resnet50(pretrained=False)
        
        # Modify the classifier head
        self.pretrained_model.fc = nn.Linear(self.pretrained_model.fc.in_features, num_classes)
        
        # Initialize the weights of the classifier head
        self._initialize_classifier_weights(self.pretrained_model.fc)

    def _initialize_classifier_weights(self, layer):
        """
        Initializes the weights of the classifier layer with Kaiming normal distribution
        and biases to zero.
        """
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        return self.pretrained_model(x)




#%%% TRENSFORMER (ViT) (from https://github.com/tintn/vision-transformer-from-scratch/blob/main/vit.py)




class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP_Vit(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP_Vit(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(ImageClassificationBase):
    """
    The ViT model for classification.
    """

    def __init__(self, config, params):
        self.params = params.copy()
        
        """        
        super(VGG, self).__init__()
        super(VGG, self).__init__(self, params['n_out'], params['NSteps'], params['n_epochs'])
        """
        nn.Module.__init__(self)
        
        #super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        
        self.prev_size=self.hidden_size
        self.n_outputs=self.num_classes
        
        self.output = nn.Linear(self.prev_size, self.n_outputs)
        # Initialize the weights
        self.apply(self.initialize_weights)

    def forward(self, x, output_attentions=False):
        outs = {}
        

        
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.output(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
                
        """
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
        """
        outs['l2'] = encoder_output[:, 0, :]

        outs['out'] = logits
        #outs['pred'] = torch.argmax(logits, dim=1)
        #return outs
        return outs['out']
        
        
    
    
    
    def initialize_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            #torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            nn.init.kaiming_normal_(module.weight) #using the same init compatible with our analisys
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)












#%%define instance of the model

            
config = {
    "patch_size": 4,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": num_classes,
    "num_channels": 3,
    "qkv_bias": False,
}














#%%MODEL INSTANCE

if Architecture == 'MLP':
    model = to_device(SimpleMLP(params), device)
elif Architecture == 'MLP-mixer':
    model = to_device(MLPMixer(image_size=32,
    patch_size=8,
    in_channels=3,
    num_features=32,
    expansion_factor=2,
    num_layers=8,
    num_classes=2,
    dropout=0.3,), device)
    
elif Architecture == 'BaseCNN':
    model = to_device(BaseCNNModel(), device)
    
elif Architecture == 'ViT':
    model = to_device(ViTForClassfication(config, params), device)
elif Architecture=='ResNet':
    model = to_device(ResNet9(3, num_classes, params), device)
    
elif Architecture=='MLP_mixer2':
    model =  to_device(MLPMixer_v2(
        params,
        in_channels=3,
        img_size=32, 
        patch_size=4, 
        hidden_size=128, 
        hidden_s=512, 
        hidden_c=64, 
        num_layers=8, 
        num_classes=num_classes, 
        drop_p=0.,
        off_act=False,
        is_cls_token=True,
        ), device)
elif Architecture=='PT_ResNet50':
    model = to_device(CustomResNet50(num_classes), device)

    
if (ds=='Gaussian' or ds == 'ImbalancedGaussian'):
    model.double()
#print(model)







#%%TRAINING
"""
 there are two possible modes: 
     TrainMode=='OFF' (collecting statistics at initializzation)
     TrainMode=='ON' (performing the whole dynamics)
     
"""
if TrainMode=='OFF':
    
    model.StoringVariablesCreation()

    #initialize the vector of guesses
    InitGuess = np.zeros(num_classes)    
    
    
    
    #init evaluation
    #TRAIN
    model.train() #set to train just to include dropout/batch norm for the evaluation measures
    
    
    bs_out = [model.validation_step(batch, num_trdata_points, 'KickStart', params) for batch in enumerate(train_dl)] #compute statistics on single batches
    ds_out = model.validation_epoch_end(bs_out, 'Train')  #group bs stat into global (DataSet one)
    


    #%%     WANDB INIT
    
    if SplittingStats=='ON':
        ProjName = 'IGB_Net_{}_Dataset_{}_IR_{}'.format(Architecture, ds, ImbalanceFactor)  #the project refers to all the simulations we would like to compare
    else:
        ProjName = 'ProvExt_Optim_{}_GradNorm_{}_IGB_Net_{}_Dataset_{}'.format(OptimFlag,GradNormMode, Architecture, ds)  #the project refers to all the simulations we would like to compare
    GroupName = '/Aug_IGB_{}_lr_{}_Bs_{}_KS_{}_DS_{}~'.format( IGB_Mode, learning_rate, batch_size, ks, shift_const) #the group identifies the simulations we would like to average togheter for the representation
    RunName = '/Sample' + str(args.SampleIndex)  #the run name identify the single run belonging to the above broad categories
    WandbId = wandb.util.generate_id()
    
    
    #we define a list of tags that we can use for group more easly runs on wandb
    #we list all the relevant parameter as tag
    tags = ["LR_{}".format(learning_rate), "BS_{}".format(batch_size), "IGB_{}".format(IGB_Mode), 'KS_{}'.format(params['ks']), 'DS_{}'.format(shift_const), 'IGB_Level_{}'.format(IGB_Sel_Mode)]
    
    run = wandb.init(project= ProjName, #PCNSGD_vs_SGD #CNN_PCNSGD_VS_SGD
               group =  GroupName,#with the group parameter you can organize runs divide them into groups
               #job_type="ImbRatio_{}_lr_{}_Bs_{}_GF_{}_DP_{}_Classes_1_9".format(UnbalanceFactor,learning_rate, batch_size, group_factor, dropout_p) , #specify a subsection inside a simulation group
               #dir = compl_wandb_dir,
               tags = tags,
               notes="experiments to compare the effect of IGB on MLP with a balanced dataset",
               entity= "emanuele_francazi", #"gpu_runs", #
               name = RunName,
               id = WandbId, #you can use id to resume the corresponding run; to do so you need also to express the resume argument properly
               resume="allow"
               #sync_tensorboard=True,
               ,reinit=True #If you're trying to start multiple runs from one script, add two things to your code: (1) run = wandb.init(reinit=True): Use this setting to allow reinitializing runs
               )
    
    
    CustomizedX_Axis() #set the customized x-axis in a automatic way for all the exported charts
    
    
    
    wandb.config = {
      "learning_rate": learning_rate,
      "epochs": epochs,
      "batch_size": batch_size,
      "architecture": Architecture,
      "Dataset":ds
    }
    
 




    
    with open("./RunsResults/GuessImbalance.txt", "a") as f:
        f.write('{}\n'.format(ds_out['val_f'][0]))
       
    with open('./RunsResults/ClassesGI.txt', "a") as f:
        np.savetxt(f, [ds_out['val_f'].numpy()], delimiter = ',')
        
    
    with open('./RunsResults/ClassesMaxf.txt', "a") as f:
        f.write('{}\n'.format(np.max(ds_out['val_f'].numpy())))





#Training the model
elif TrainMode=='ON':
    

    #%% PRELIMINARY STATS
    
    #here we want to estimate the level of IGB (useful when we don't collect stat at initialization, i.e. when we do training mode)
    
    
    N_Init=50
    Output_Stats = []
    if SigmasStatsFlag=='ON':
        sigma_stats = InitVarComp(N_Init, Output_Stats, train_dl, params, N_Init, model)
    
    
    #here an estimate for the level of IGB associated with the current initialization (that we will use for the training)
    
    Sigma_Single_Stats = SingleInitStats(train_dl, params, model)
    
        
    
    InitStat_checkPoint = time.time()

    InitStat_Time_diff = InitStat_checkPoint - Load_checkPoint 

    print('TIME from data loading till computation of init stats is: ', InitStat_Time_diff)
    
    
    #%% PREPARING FOR TRAINING
    
    
    model.StoringVariablesCreation()
    
    history = []
    
    
    
    #init evaluation
    #TRAIN
    model.train() #set to train just to include dropout/batch norm for the evaluation measures
    
    
    
    
    
    
    """
    #specify the path for label and figure
    label_path= './label1.txt'
    figure_path='./figure1.png'
    
    save_image_and_label(train_ds[0], figure_path, label_path)
    """
    
    
    #REASSIGN THE LABELS ACCORDING TO THE IGB
    if OrderingClassesFlag=='ON':
        
        bs_out = [model.validation_step( batch, num_trdata_points, 'Train', params) for batch in enumerate(train_dl)] #compute statistics on single batches
        ds_out = model.validation_epoch_end(bs_out, 'Train')  #group bs stat into global (DataSet one)

        print("params['label_list']:", params['label_list'])
        print("Type of params['label_list']:", type(params['label_list']))
        for item in params['label_list']:
            print(f"Item: {item}, Type: {type(item)}")


        label_map = label_map_function(ds_out['val_f'], params) #first defining the dict of mapping based on the initial fractions of guesses
        params['label_map']=label_map
        
        
        print('fractions before the swap are', ds_out['val_f'])#print the fractions before the swap of labels
        print('the mapping we will perform is', label_map)
        
        OrderOutputNodes(model, params)
        if (ds=='Gaussian' or ds == 'ImbalancedGaussian'):
            model.double()  #recast the model as double because (biases of the new output may be float as default)
        
        
        #reset the variables computed with the initial order of labels (a more efficient choice would be to swap the per-class statistics already computed)
        model.StoringVariablesCreation()
        
    
    """
    #CHANGE LABELS ACCORDING TO IGB
    #create now a deep copy of targets so that we will keep an external reference during the map; otherwise we can have problem:
        #for example let us consider the binary case, when we want to swap the 2 classes. if we map 0 in 1 then all the targets become ==1; if after this we give the command to map the 1 in 0s we do not map only the inital targets with value==1 but also the remaining (original 0s) mapped in the previous step
    #copying list of targets
    traintargets = torch.tensor(train_ds.targets)
    validtargets = torch.tensor(valid_ds.targets)
    
    train_ds.targets = torch.tensor(train_ds.targets)
    valid_ds.targets = torch.tensor(valid_ds.targets)
    
    
    for key in label_map:               
        train_ds.targets[traintargets==key]=label_map[key] 
        valid_ds.targets[validtargets==key]=label_map[key]
    
    #check images and labels after
    
    #specify the path for label and figure
    label_path= './label2.txt'
    figure_path='./figure2.png'
    
    save_image_and_label(train_ds[0], figure_path, label_path)
    
    #redefine dataloaders with the new labels
    
    # PyTorch data loaders
    if ds=='Gaussian':
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size*2, shuffle=True) 
    else:
        #train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True) #use this to check if you get the same observable in case of no label swaps (as sanity check)
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
        valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)
    
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    """
    
    
    """
    if Loss_function=='Hinge':
        #print('targets before', dict(Counter(train_ds.targets.tolist())))
        
        #copying list of targets
        traintargets = copy.deepcopy(train_ds.targets)
        validtargets = copy.deepcopy(valid_ds.targets)
        
        for key in label_map:
            train_ds.targets[traintargets==label_map[key]]= key
            valid_ds.targets[validtargets==label_map[key]]= key
        
        #print('targets after', dict(Counter(train_ds.targets.tolist())))
            
    else:
        OrderOutputNodes(model, label_map)
        model.double() #recast the model as double because (biases of the new output may be float as default)
    """
    
    
    
    
    #now repeat the propagation of the train dataset
    
    
    
    bs_out = [model.validation_step(batch, num_trdata_points, 'KickStart', params) for batch in enumerate(train_dl)] #compute statistics on single batches
    ds_out = model.validation_epoch_end(bs_out, 'Train')  #group bs stat into global (DataSet one)
    
    
    # now let's print the GradNorm and the initial stepsize 
    print('average of batch grad norm after init, for our model with dataset shift {} , is {} while the step size {}'.format(shift_const , model.GradNorm[0], model.GradNorm[0]*learning_rate))
    
    
    
    if FixedStepSizeFlag=='ON':
        #we set now the learning rate to a value such that the starting step size is 5*10^-5 (value chosen to stay in the slow regime)
        learning_rate = FixStepSize/model.GradNorm[0]
        print('the learning rate update value is {} ; this had been chosen to fix the starting step size to the value {}, compatible with the average Grad Norm {}'.format(learning_rate, FixStepSize, model.GradNorm[0]))
    
    #append the initial StepSize:
    model.StepSize.append(model.GradNorm[0]*learning_rate)    
        
    
    print('fractions after the swap are', ds_out['val_f'])#print the fractions before the swap of labels
    
    
    
   
    
if IGB_Sel_criterion=='Fractions':

    #IGB_Value = abs(ds_out['val_f'][0] - 0.5)
    IGB_Diff= np.max(ds_out['val_f'].numpy()) - np.min(ds_out['val_f'].numpy())
    print('IGB vector ', ds_out['val_f'], ' and diff ', IGB_Diff)


    if IGB_Sel_Mode=='Low':
        if IGB_Diff>0.10: #if we want to select the samples with small IGB
            print('Sample ', args.SampleIndex, 'has too big imbalance')
            sys.exit()  #If the imbalance is too high Terminate the program with status code 0
        
    elif IGB_Sel_Mode=='High':      
        if ((IGB_Diff < 0.90 )): #if we want to select the samples with High IGB
            print('Sample ', args.SampleIndex, 'has too little imbalance: ', ds_out['val_f'][0].item())
            sys.exit()  #If the imbalance is too high Terminate the program with status code 0

elif IGB_Sel_criterion=='GammaProxy':
    ratio = Sigma_Single_Stats['output_centers_distance'] / Sigma_Single_Stats['mean_output_variance']
    print('Ratio:', ratio)

    if IGB_Sel_Mode == 'Low':
        if ratio > 1:  # if we want to select the samples with small IGB
            print('Sample', args.SampleIndex, 'has too big imbalance')
            sys.exit()  # If the imbalance is too high, terminate the program with status code 0

    elif IGB_Sel_Mode == 'High':
        if ratio < 30:  # if we want to select the samples with High IGB
            print('Sample', args.SampleIndex, 'has too little imbalance:', ratio)
            sys.exit()  # If the imbalance is too high, terminate the program with status code 0

    
    
    #%% creation of data folder
    
    
    
    #we first create the folder associated to the sample and then save inside the folder all the files
    #we start by creating the path for the folder to be created
    #we first create the parameter folder
    
    if SplittingStats=='ON':
        print('here the max class is: ', np.argmax(ds_out['val_f'].numpy()), ds_out['val_f'].numpy())
        
        FolderPath = './RunsResults/'+ args.FolderName + '/IGB' + str(np.argmax(ds_out['val_f'].numpy()))
    else:
        FolderPath = './RunsResults/'+ args.FolderName
    if not os.path.exists(FolderPath):
        os.makedirs(FolderPath, exist_ok=True)         
    #then we create the specific sample folder
    FolderPath = FolderPath +'/LR' + str(args.lr) + '/KS' + str(args.ks) +  '/Slope' + str(args.Relu_Slope) + '/Data_Shift' + str(shift_const) + '/Sample' + str(args.SampleIndex)
    print('La cartella creata per il sample ha come path: ', FolderPath)
    if not os.path.exists(FolderPath):
        os.makedirs(FolderPath, exist_ok=True) 
    
    params['FolderPath'] = FolderPath #add a new param to the param dict
    
    
    
    
    
    
    
    #%%     WANDB INIT
    
    if SplittingStats=='ON':
        ProjName = 'ImbCD_{}_GradNorm_IGB_Net_{}_Dataset_{}_IR_{}'.format(np.argmax(ds_out['val_f'].numpy()), Architecture, ds, ImbalanceFactor)  #the project refers to all the simulations we would like to compare
    else:
        ProjName = 'Optim_{}_GradNorm_{}_IGB_Net_{}_Dataset_{}'.format(OptimFlag,GradNormMode, Architecture, ds)  #the project refers to all the simulations we would like to compare
    GroupName = '/IGBLev_{}_lr_{}_Bs_{}_KS_{}_DS_{}~'.format(IGB_Sel_Mode, learning_rate, batch_size, ks, shift_const) #the group identifies the simulations we would like to average togheter for the representation
    RunName = '/Sample' + str(args.SampleIndex)   #the run name identify the single run belonging to the above broad categories
    WandbId = wandb.util.generate_id()
    
    
    #we define a list of tags that we can use for group more easly runs on wandb
    #we list all the relevant parameter as tag
    tags = ["LR_{}".format(learning_rate), "BS_{}".format(batch_size), "IGB_{}".format(IGB_Sel_Mode), 'KS_{}'.format(params['ks']), 'DS_{}'.format(shift_const)]
    
    run = wandb.init(project= ProjName, #PCNSGD_vs_SGD #CNN_PCNSGD_VS_SGD
               group =  GroupName,#with the group parameter you can organize runs divide them into groups
               #job_type="ImbRatio_{}_lr_{}_Bs_{}_GF_{}_DP_{}_Classes_1_9".format(UnbalanceFactor,learning_rate, batch_size, group_factor, dropout_p) , #specify a subsection inside a simulation group
               #dir = compl_wandb_dir,
               tags = tags,
               notes="experiments to compare the effect of IGB on MLP with a balanced dataset",
               entity= "emanuele_francazi", #"gpu_runs", #
               name = RunName,
               id = WandbId, #you can use id to resume the corresponding run; to do so you need also to express the resume argument properly
               resume="allow"
               #sync_tensorboard=True,
               ,reinit=True #If you're trying to start multiple runs from one script, add two things to your code: (1) run = wandb.init(reinit=True): Use this setting to allow reinitializing runs
               )
    
    
    CustomizedX_Axis() #set the customized x-axis in a automatic way for all the exported charts
    
    
    
    wandb.config = {
      "learning_rate": learning_rate,
      "epochs": epochs,
      "batch_size": batch_size,
      "architecture": Architecture,
      "Dataset":ds
    }
    
    
    
    
    
    
    
    
    
    
    
    #%%saving statistics
    
    #Computed over many instances
    if SigmasStatsFlag=='ON':
        # Convert the dictionary values to a 1D NumPy array
        stats_array = np.array([sigma_stats[key] for key in sigma_stats])
        
        # Reshape to ensure it's a 1D array
        stats_array = np.reshape(stats_array, (-1,))  # Convert to 1D array
        
        #saving the sigma stas (for IGB quantification) on file
        with open(params['FolderPath'] + "/Sigmas.txt", "w") as f:
            np.savetxt(f, stats_array, delimiter = ',') 
            
        sigmas_ratio = sigma_stats['ensemble_centers_std']/sigma_stats['dataset_output_std']
        # Saving the ratio to a file as a NumPy array
        np.savetxt(params['FolderPath'] + "/Sigmas_Ratio.txt", np.array([sigmas_ratio]), fmt='%.3f')
    
    
    #Computed over the single (current) instance
    
    # Convert the dictionary values to a 1D NumPy array
    Single_stats_array = np.array([Sigma_Single_Stats[key] for key in Sigma_Single_Stats])
    
    # Reshape to ensure it's a 1D array
    Single_stats_array = np.reshape(Single_stats_array, (-1,))  # Convert to 1D array
    
    #saving the sigma stas (for IGB quantification) on file
    with open(params['FolderPath'] + "/Sigmas.txt", "w") as f:
        np.savetxt(f, Single_stats_array, delimiter = ',') 
        
    Single_sigmas_ratio = Sigma_Single_Stats['output_centers_distance'] / Sigma_Single_Stats['mean_output_variance']
    # Saving the ratio to a file as a NumPy array
    np.savetxt(params['FolderPath'] + "/SingleInstance_Sigmas_Ratio.txt", np.array([Single_sigmas_ratio]), fmt='%.3f')
    

    
    
    
    
    
    
    
    with open("./RunsResults/TrainGuessImbalance.txt", "a") as f:
        f.write('{}\n'.format(ds_out['val_f'][0]))
       
    with open('./RunsResults/TrainClassesGI.txt', "a") as f:
        np.savetxt(f, [ds_out['val_f'].numpy()], delimiter = ',')
        
    
    with open('./RunsResults/TrainClassesMaxf.txt', "a") as f:
        f.write('{}\n'.format(np.max(ds_out['val_f'].numpy())))
    
    #TEST
    
    model.eval()
    
    bs_out = [model.validation_step(batch, num_valdata_points, 'Eval', params) for batch in enumerate(valid_dl)] #compute statistics on single batches
    ds_out = model.validation_epoch_end(bs_out, 'Eval')  #group bs stat into global (DataSet one)
    
    #first log on wandb
    WandB_logs(0, model)
    
    
    with open("./RunsResults/TestGuessImbalance.txt", "a") as f:
        f.write('{}\n'.format(ds_out['val_f'][0]))
       
    with open('./RunsResults/TestClassesGI.txt', "a") as f:
        np.savetxt(f, [ds_out['val_f'].numpy()], delimiter = ',')
        
    
    with open('./RunsResults/TestClassesMaxf.txt', "a") as f:
        f.write('{}\n'.format(np.max(ds_out['val_f'].numpy())))
    
    model.time.append(0)
    
    save_on_file(model, params)
        
        
        
          
    
    
    
    
    
    
    
    
    #%%Training
    
    
    history += fit_one_cycle(epochs, TimeValSteps,learning_rate, model, train_dl, valid_dl, params,
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
    
else:
    print('TrainMode Flag set incorrectly; please choose either ON or OFF as value')

#print(history)

with open("./RunsResults/ExecTimes.txt", "a") as f:
    f.write('{}\n'.format(time.time() - start_TotTime))











