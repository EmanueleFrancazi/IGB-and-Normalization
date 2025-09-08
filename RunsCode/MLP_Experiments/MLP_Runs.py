import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb  # make sure to install wandb (pip install wandb)
import itertools
from torch.utils.data import Subset
import torchvision
print('torchvision version', torchvision.__version__, flush=True)
import torchvision.transforms as transforms


from utils.IGB_utils import *

import random

from datasets import load_dataset
from torch.utils.data import Dataset

import torchvision.transforms as transforms

# Print the process ID
print(f"Process ID: {os.getpid()}")

#%%% CUSTOMIZE THE AXIS ASSOCIATED TO LOGGED METRICS (WANDB)
def CustomizedX_Axis():
    """
    Set the default x axis for wandb charts:
    All metrics under Performance_measures/* will use the step given by Performance_measures/True_Steps_+_1.
    """
    wandb.define_metric("Performance_measures/True_Steps_+_1")
    wandb.define_metric("Performance_measures/*", step_metric="Performance_measures/True_Steps_+_1")
    wandb.define_metric("Check/Epoch")
    wandb.define_metric("Check/*", step_metric="Check/Epoch")

#############################################
# 1. Data generation: two Gaussian blobs
#############################################
def generate_gaussian_blobs(n_samples, dim, center_val, sigma2, device):
    """
    Generate two Gaussian blobs.
    If center_val is a scalar m then one blob is centered at [m, m, ..., m]
    and the other at [-m, -m, ..., -m]. Covariance is sigma^2 * I.
    """
    sigma = np.sqrt(sigma2)
    center1 = torch.full((dim,), center_val, device=device)
    center2 = torch.full((dim,), -center_val, device=device)
    
    X1 = center1 + sigma * torch.randn(n_samples, dim, device=device)
    X2 = center2 + sigma * torch.randn(n_samples, dim, device=device)
    Y1 = torch.zeros(n_samples, dtype=torch.long, device=device)  # label 0
    Y2 = torch.ones(n_samples, dtype=torch.long, device=device)   # label 1
    
    X = torch.cat([X1, X2], dim=0)
    Y = torch.cat([Y1, Y2], dim=0)
    perm = torch.randperm(X.size(0))
    return X[perm], Y[perm]



class TinyImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        # This line creates a .targets attribute as a list of int labels
        self.targets = [item['label'] for item in hf_dataset]

    def __getitem__(self, idx):
        img = self.hf_dataset[idx]['image']
        label = self.hf_dataset[idx]['label']
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.hf_dataset)



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

"""
# this version allow to select the number of datapoints for each label: need to debug

def get_dataset_and_input_dim(param_config, device, train=True):
    dataset_name = param_config.get("dataset", "Gaussian").lower()
    offset_value = param_config.get("offset_value", 0.0)
    model_name = param_config.get('model', 'MLP')
    n_per_class = param_config.get("n_per_class", None)
    model_name = param_config.get('model', 'MLP')

    if dataset_name == "gaussian":
        n_samples = 10000 if train else 500
        input_dim = 1000
        center_val = 1.0 / np.sqrt(input_dim)
        sigma2 = 1.0
        X, Y = generate_gaussian_blobs(n_samples, input_dim, center_val, sigma2, device)
        dataset = TensorDataset(X, Y)
    elif dataset_name == "mnist":
        if model_name == 'MLP':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x + offset_value),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
            input_dim = 28 * 28
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + offset_value),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
            ])
            dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
            input_dim = None
    elif dataset_name == "cifar10":
        if model_name == 'MLP':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: x + offset_value),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
            input_dim = 32 * 32 * 3
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + offset_value),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
            input_dim = None
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized!")
        
    # --- Apply class filtering/aggregation if specified ---
    class_mapping = param_config.get("class_mapping", None)
    if class_mapping is not None:
        dataset = filter_dataset_by_class_mapping(dataset, class_mapping, remap=True, n_per_class=n_per_class)
        
    return dataset, input_dim




def filter_dataset_by_class_mapping(dataset, class_mapping, remap=True, n_per_class=None):
    
    # Filters a dataset based on a provided class_mapping dictionary and subsamples each class.
    
    # Args:
    #   dataset: The dataset to filter (supports torchvision or TensorDataset).
    #   class_mapping: A dict mapping original labels to new labels.
    #   remap: If True, remap the labels as specified.
    #   n_per_class: Controls subsampling. It can be:
    #       - None: Use all samples.
    #       - int: Use at most n_per_class samples from each label (if available).
    #       - dict: A mapping from each original label to the desired number of samples.
    #       - 'min': Automatically select the minimum number of samples available across all specified classes.
      
    # Returns:
    #   A new dataset with filtered, optionally remapped, and subsampled labels.
    
    if class_mapping is None:
        return dataset  # No filtering
    
    filter_labels = set(class_mapping.keys())
    indices_by_label = {label: [] for label in filter_labels}
    
    # Retrieve labels from dataset (supporting both torchvision and TensorDataset)
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if isinstance(targets, list):
            for i, label in enumerate(targets):
                if label in filter_labels:
                    indices_by_label[label].append(i)
        else:
            for c in filter_labels:
                idx = (targets == c).nonzero(as_tuple=True)[0].tolist()
                indices_by_label[c] = idx
    elif hasattr(dataset, 'tensors'):
        labels = dataset.tensors[1]
        for c in filter_labels:
            idx = (labels == c).nonzero(as_tuple=True)[0].tolist()
            indices_by_label[c] = idx
    else:
        raise ValueError("Dataset type not supported for filtering.")
    
    #DEBUG
    for label, idx_list in indices_by_label.items():
        print(f"Label {label}: {len(idx_list)} samples")    

    
    # If the user wants an automated balance, compute the minimum count once.
    if n_per_class == 'min':
        min_count = min(len(idx) for idx in indices_by_label.values())

        #DEBUG
        print(f"Computed min_count: {min_count}")

    selected_indices = []

    #DEBUG
    selected_count = {label: 0 for label in indices_by_label.keys()}

    for label, indices in indices_by_label.items():
        # Determine number of samples to select for this label
        if n_per_class is None:
            n = len(indices)
        elif isinstance(n_per_class, dict):
            n = min(n_per_class.get(label, len(indices)), len(indices))
        elif isinstance(n_per_class, int):
            n = min(n_per_class, len(indices))
        elif n_per_class == 'min':
            n = min_count
        else:
            raise ValueError("Invalid value for n_per_class.", n_per_class)
        
        # Shuffle indices to ensure randomness then take the first n
        random.shuffle(indices)            # Uncomment if you want to shuffle the indices, i.e. having different samples each run.
        chosen = indices[:n]
        selected_count[label] += len(chosen)
        selected_indices.extend(chosen)


    print("Selected counts per label:", selected_count)
    print("Total selected samples:", len(selected_indices))



    # Create a subset using the selected indices
    subset = Subset(dataset, selected_indices)
    
    if remap:
        # Define a remapping dataset that applies the mapping on-the-fly.
        class RemappedDataset(torch.utils.data.Dataset):
            def __init__(self, subset, mapping):
                self.subset = subset
                self.mapping = mapping
            def __getitem__(self, index):
                data, label = self.subset[index]
                # Convert label to a Python int if it's a tensor.
                if isinstance(label, torch.Tensor):
                    label = label.item()
                return data, self.mapping[label]
            def __len__(self):
                return len(self.subset)
        return RemappedDataset(subset, class_mapping)
    else:
        return subset
"""

def get_dataset_and_input_dim(param_config, device, train=True):
    dataset_name = param_config.get("dataset", "Gaussian").lower()
    offset_value = param_config.get("offset_value", 0.0)
    
    if dataset_name == "gaussian":
        n_samples = 10000 if train else 500
        input_dim = 1000
        center_val = 1.0 / np.sqrt(input_dim)
        sigma2 = 1.0
        X, Y = generate_gaussian_blobs(n_samples, input_dim, center_val, sigma2, device)
        dataset = TensorDataset(X, Y)
    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x + offset_value),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        input_dim = 28 * 28
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x + offset_value),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        input_dim = 32 * 32 * 3

    elif dataset_name == "tinyimagenet":
        from datasets import load_dataset
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Lambda(lambda x: x + offset_value),
            #transforms.Lambda(lambda x: x.view(-1))  # Remove for ViT!
        ])
        hf_dataset = load_dataset("zh-plus/tiny-imagenet") # from https://huggingface.co/datasets/zh-plus/tiny-imagenet (other datasets in https://huggingface.co/datasets?sort=trending&search=imagenet)
        split = "train" if train else "valid"
        dataset = TinyImageNetDataset(hf_dataset[split], transform=transform)
        input_dim = 64 * 64 * 3

    else:
        raise ValueError(f"Dataset {dataset_name} not recognized!")
        
    # --- Apply class filtering/aggregation if specified ---
    class_mapping = param_config.get("class_mapping", None)
    if class_mapping is not None:
        dataset = filter_dataset_by_class_mapping(dataset, class_mapping, remap=True)
        
    return dataset, input_dim


def filter_dataset_by_class_mapping(dataset, class_mapping, remap=True):
    """
    Filters a dataset based on a provided class_mapping dictionary.
    
    Args:
      dataset: The dataset to filter (supports torchvision or TensorDataset).
      class_mapping: A dict mapping original labels to new labels.
      remap: If True, remap the labels as specified.
      
    Returns:
      A new dataset with filtered and remapped labels.
    """
    if class_mapping is None:
        return dataset  # No filtering
    
    filter_labels = set(class_mapping.keys())
    indices = []
    
    # Retrieve labels from dataset (supporting both torchvision and TensorDataset)
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if isinstance(targets, list):
            for i, label in enumerate(targets):
                if label in filter_labels:
                    indices.append(i)
        else:
            indices = (torch.stack([targets == c for c in filter_labels]).any(dim=0)).nonzero(as_tuple=True)[0].tolist()
    elif hasattr(dataset, 'tensors'):
        labels = dataset.tensors[1]
        indices = (torch.stack([labels == c for c in filter_labels]).any(dim=0)).nonzero(as_tuple=True)[0].tolist()
    else:
        raise ValueError("Dataset type not supported for filtering.")
    
    subset = Subset(dataset, indices)
    
    if remap:
        # Define a simple remapping dataset that applies the mapping on-the-fly.
        class RemappedDataset(torch.utils.data.Dataset):
            def __init__(self, subset, mapping):
                self.subset = subset
                self.mapping = mapping
            def __getitem__(self, index):
                data, label = self.subset[index]
                return data, self.mapping[label]
            def __len__(self):
                return len(self.subset)
        return RemappedDataset(subset, class_mapping)
    else:
        return subset


#############################################
# Swin Transformer definition
#############################################
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

try:
    from timm.models.layers import trunc_normal_, DropPath
except Exception:
    from torch.nn.init import trunc_normal_
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            return x

try:
    from thop import profile  # optional; used only if available
except Exception:
    profile = None

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(w, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output
    
    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0], relation[:,:,1]]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None, norm_config='ln_before'):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )
        self.norm_config = norm_config or 'ln_before'

    def forward(self, x):
        if self.norm_config == 'ln_before':
            x = x + self.drop_path(self.msa(self.ln1(x)))
            x = x + self.drop_path(self.mlp(self.ln2(x)))
            return x
        elif self.norm_config == 'ln_after':
            x = x + self.drop_path(self.msa(x))
            x = self.ln1(x)
            x = x + self.drop_path(self.mlp(x))
            x = self.ln2(x)
            return x
        else:
            raise ValueError("SwinTransformer supports norm_config 'ln_before' or 'ln_after' only.")

class SwinTransformer(nn.Module):
    """ Implementation of Swin Transformer https://arxiv.org/abs/2103.14030
    In this Implementation, the standard shape of data is (b h w c), which is a similar protocal as cnn.
    """
    #TODO make layers using configs
    def __init__(self, num_classes, config=[2,2,6,2], dim=96, drop_path_rate=0.2, input_resolution=224, norm_config='ln_before'):
        super(SwinTransformer, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 7
        # self.patch_partition = Rearrange('b c (h1 sub_h) (w1 sub_w) -> b h1 w1 (c sub_h sub_w)', sub_h=4, sub_w=4)

        self.norm_config = norm_config or 'ln_before'

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        begin = 0
        self.stage1 = [nn.Conv2d(3, dim, kernel_size=4, stride=4),
                       Rearrange('b c h w -> b h w c'),
                       nn.LayerNorm(dim),] + \
                      [Block(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//4, norm_config=self.norm_config) 
                      for i in range(config[0])]
        begin += config[0]
        self.stage2 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(4*dim), nn.Linear(4*dim, 2*dim, bias=False),] + \
                      [Block(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8, norm_config=self.norm_config)
                      for i in range(config[1])]
        begin += config[1]
        self.stage3 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(8*dim), nn.Linear(8*dim, 4*dim, bias=False),] + \
                      [Block(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//16, norm_config=self.norm_config)
                      for i in range(config[2])]
        begin += config[2]
        self.stage4 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(16*dim), nn.Linear(16*dim, 8*dim, bias=False),] + \
                      [Block(8*dim, 8*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//32, norm_config=self.norm_config)
                      for i in range(config[3])]
        
        self.stage1 = nn.Sequential(*self.stage1)
        self.stage2 = nn.Sequential(*self.stage2)
        self.stage3 = nn.Sequential(*self.stage3)
        self.stage4 = nn.Sequential(*self.stage4)

        self.norm_last = nn.LayerNorm(dim * 8)
        self.mean_pool = Reduce('b h w c -> b c', reduction='mean')
        self.classifier = nn.Linear(8*dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.norm_last(x)

        x = self.mean_pool(x)
        x = self.classifier(x)
        return x

def Swin_T(num_classes, config=[2,2,6,2], dim=96, input_resolution=224, norm_config='ln_before', **kwargs):
    return SwinTransformer(num_classes, config=config, dim=dim, input_resolution=input_resolution, norm_config=norm_config, **kwargs)

def Swin_S(num_classes, config=[2,2,18,2], dim=96, input_resolution=224, norm_config='ln_before', **kwargs):
    return SwinTransformer(num_classes, config=config, dim=dim, input_resolution=input_resolution, norm_config=norm_config, **kwargs)

def Swin_B(num_classes, config=[2,2,18,2], dim=128, input_resolution=224, norm_config='ln_before', **kwargs):
    return SwinTransformer(num_classes, config=config, dim=dim, input_resolution=input_resolution, norm_config=norm_config, **kwargs)

def Swin_L(num_classes, config=[2,2,18,2], dim=192, input_resolution=224, norm_config='ln_before', **kwargs):
    return SwinTransformer(num_classes, config=config, dim=dim, input_resolution=input_resolution, norm_config=norm_config, **kwargs)

#############################################
# 3. Filtering mode: check initial imbalance
#############################################
def filtering_check(model, data, device):
    """
    Run a forward pass (without gradients) on the dataset and compute the fraction
    of datapoints predicted as each class.
    Returns:
        diff: absolute difference between most and least guessed class fractions
        frac0: fraction of datapoints predicted as class 0
        frac1: fraction of datapoints predicted as class 1 (if exists, else None)
        class_fractions: list of fractions for all classes
        max_fraction: maximum among class fractions
    """
    with torch.no_grad():
        outputs = model(data)
        preds = outputs.argmax(dim=1)
        num_classes = int(outputs.shape[1])
        class_fractions = []
        total = preds.numel()
        for cls in range(num_classes):
            frac = (preds == cls).float().mean().item()
            class_fractions.append(frac)
        max_fraction = max(class_fractions)
        min_fraction = min(class_fractions)
        # For backward compatibility
        frac0 = class_fractions[0]
        frac1 = class_fractions[1] if num_classes > 1 else None
        diff = abs(frac0 - frac1)

    return diff, frac0, frac1, class_fractions, max_fraction

#############################################
# 4. Evaluation: compute loss and accuracy (global and per class)
#############################################
def evaluate_dataset(model, dataset, criterion, device, set_type='test', eval_batch_size=128):
    """
    Evaluate the model on a dataset.
    
    Returns a dictionary:
       'global': {loss, accuracy, frac0, frac1},
       0: {loss, accuracy} for class 0,
       1: {loss, accuracy} for class 1.
    """
    if eval_batch_size is None:
        data_loader = [(dataset.tensors[0].to(device), dataset.tensors[1].to(device))]
    else:
        data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    
    if set_type == 'train':
        model.train()
    else:
        model.eval()
    
    total_loss, total_correct, total_samples = 0.0, 0, 0
    loss_class0, loss_class1 = 0.0, 0.0
    correct_class0, correct_class1 = 0, 0
    count_class0, count_class1 = 0, 0
    guess_count0, guess_count1 = 0, 0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            losses = criterion(outputs, y_batch)
            preds = outputs.argmax(dim=1)
            
            bs = y_batch.size(0)
            total_loss += losses.sum().item()
            total_correct += (preds == y_batch).sum().item()
            total_samples += bs
            for cls in [0, 1]:
                mask = (y_batch == cls)
                n_cls = mask.sum().item()
                if n_cls > 0:
                    if cls == 0:
                        loss_class0 += losses[mask].sum().item()
                        correct_class0 += (preds[mask] == cls).sum().item()
                        count_class0 += n_cls
                    else:
                        loss_class1 += losses[mask].sum().item()
                        correct_class1 += (preds[mask] == cls).sum().item()
                        count_class1 += n_cls
            guess_count0 += (preds == 0).sum().item()
            guess_count1 += (preds == 1).sum().item()
    
    global_loss = total_loss / total_samples
    global_acc = total_correct / total_samples
    frac0 = guess_count0 / total_samples
    frac1 = guess_count1 / total_samples
    class0_loss = loss_class0 / count_class0 if count_class0 > 0 else None
    class0_acc = correct_class0 / count_class0 if count_class0 > 0 else None
    class1_loss = loss_class1 / count_class1 if count_class1 > 0 else None
    class1_acc = correct_class1 / count_class1 if count_class1 > 0 else None

    metrics = {
        'global': {'loss': global_loss, 'accuracy': global_acc, 'frac0': frac0, 'frac1': frac1},
        0: {'loss': class0_loss, 'accuracy': class0_acc},
        1: {'loss': class1_loss, 'accuracy': class1_acc}
    }
    return metrics

#############################################
# 5. Logging: write each metric to a separate txt file
#############################################
def log_value(log_dir, file_name, value, include_step=False, step=None):
    """
    Append a line with the given value to the file.
    For metric files, include_step is False.
    """
    if include_step and step is not None:
        text = f"{step} {value}\n"
    else:
        text = f"{value}\n"
    with open(os.path.join(log_dir, file_name), 'a') as f:
         f.write(text)

def log_metrics(log_dir, step, train_metrics, test_metrics):
    # Log evaluation step in a separate file (include step)
    log_value(log_dir, 'eval_times.txt', step, include_step=True, step=step)
    # --- Training metrics (only values) ---
    log_value(log_dir, 'train_global_loss.txt', train_metrics['global']['loss'])
    log_value(log_dir, 'train_global_accuracy.txt', train_metrics['global']['accuracy'])
    log_value(log_dir, 'train_frac0.txt', train_metrics['global']['frac0'])
    log_value(log_dir, 'train_frac1.txt', train_metrics['global']['frac1'])
    train_max_frac = max(train_metrics['global']['frac0'], train_metrics['global']['frac1'])
    log_value(log_dir, 'train_max_frac.txt', train_max_frac)
    log_value(log_dir, 'train_class0_loss.txt', train_metrics[0]['loss'])
    log_value(log_dir, 'train_class0_accuracy.txt', train_metrics[0]['accuracy'])
    log_value(log_dir, 'train_class1_loss.txt', train_metrics[1]['loss'])
    log_value(log_dir, 'train_class1_accuracy.txt', train_metrics[1]['accuracy'])
    # --- Test metrics (only values) ---
    log_value(log_dir, 'test_global_loss.txt', test_metrics['global']['loss'])
    log_value(log_dir, 'test_global_accuracy.txt', test_metrics['global']['accuracy'])
    log_value(log_dir, 'test_frac0.txt', test_metrics['global']['frac0'])
    log_value(log_dir, 'test_frac1.txt', test_metrics['global']['frac1'])
    test_max_frac = max(test_metrics['global']['frac0'], test_metrics['global']['frac1'])
    log_value(log_dir, 'test_max_frac.txt', test_max_frac)
    log_value(log_dir, 'test_class0_loss.txt', test_metrics[0]['loss'])
    log_value(log_dir, 'test_class0_accuracy.txt', test_metrics[0]['accuracy'])
    log_value(log_dir, 'test_class1_loss.txt', test_metrics[1]['loss'])
    log_value(log_dir, 'test_class1_accuracy.txt', test_metrics[1]['accuracy'])

def log_ordered_metrics(log_dir, step, ordered_train_metrics, ordered_test_metrics):
    log_value(log_dir, 'train_ordered_loss_class0.txt', ordered_train_metrics[0]['loss'])
    log_value(log_dir, 'train_ordered_accuracy_class0.txt', ordered_train_metrics[0]['accuracy'])
    log_value(log_dir, 'train_ordered_loss_class1.txt', ordered_train_metrics[1]['loss'])
    log_value(log_dir, 'train_ordered_accuracy_class1.txt', ordered_train_metrics[1]['accuracy'])
    log_value(log_dir, 'test_ordered_loss_class0.txt', ordered_test_metrics[0]['loss'])
    log_value(log_dir, 'test_ordered_accuracy_class0.txt', ordered_test_metrics[0]['accuracy'])
    log_value(log_dir, 'test_ordered_loss_class1.txt', ordered_test_metrics[1]['loss'])
    log_value(log_dir, 'test_ordered_accuracy_class1.txt', ordered_test_metrics[1]['accuracy'])


def label_map_function(frac0, frac1):
    """
    Determine whether the output nodes should be swapped based on the initial guess fractions.

    If fraction of guesses assigned to node 0 is already >= 0.5, no change needed.
    If fraction assigned to node 1 is greater, nodes should be swapped.

    Returns a dictionary indicating the label map.
    """
    if frac0 >= 0.5:
        return {0: 0, 1: 1}  # No swap needed
    else:
        return {0: 1, 1: 0}  # Swap nodes


def OrderOutputNodes(model, label_map): #NOTE: this function is currently not used in the code as the new solution is to keep track of the ordering and use the label_map_function to log both raw and ordered metrics.
    """
    Reorder the weights of the output nodes according to the provided label_map.
    """
    original_weights = model.output_layer.weight.data.clone()
    original_bias = model.output_layer.bias.data.clone()

    # Initialize new weights and bias tensors
    new_weights = torch.zeros_like(original_weights)
    new_bias = torch.zeros_like(original_bias)

    # Reassign weights and biases based on label map
    for new_idx, old_idx in label_map.items():
        new_weights[new_idx] = original_weights[old_idx]
        new_bias[new_idx] = original_bias[old_idx]

    # Replace the weights and biases in the existing output layer
    model.output_layer.weight.data = new_weights
    model.output_layer.bias.data = new_bias



def get_normalized_parameters(model):
    """
    Extract all trainable parameters from the model (including weights and biases),
    flatten them into a single vector, and normalize the vector to have L2 norm = 1.
    
    Returns the normalized parameter vector.
    """
    param_list = []
    for param in model.parameters():
        if param.requires_grad:
            param_list.append(param.view(-1))
    if not param_list:
        return torch.tensor([])
    p = torch.cat(param_list)
    norm = torch.norm(p)
    if norm > 0:
        return p / norm
    else:
        return p
















#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MODELS

#############################################
# 2. MLP definition with normalization options
#############################################
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, output_dim, norm_config):
        """
        norm_config must be one of:
         - 'bn_before': BatchNorm1d before ReLU
         - 'bn_after':  BatchNorm1d after ReLU
         - 'ln_before': LayerNorm before ReLU
         - 'ln_after':  LayerNorm after ReLU
        """
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if norm_config.lower() == 'bn_before':
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
            elif norm_config.lower() == 'bn_after':
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm_config.lower() == 'ln_before':
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
            elif norm_config.lower() == 'ln_after':
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(hidden_dim))
            else:
                layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        """Initialize all linear layers with Kaiming normal and biases with zeros."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.hidden(x)
        return self.output_layer(x)
    


#############################################
# 2. ViT definition with normalization options
#############################################



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

    def __init__(self, config, params):
        super().__init__()
        self.params = params.copy()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.ln = nn.LayerNorm(config["intermediate_size"])

    def forward(self, x):
        x = self.dense_1(x)
        if self.params['norm_config']=='ln_before':
            x = self.ln(x)
        x = self.activation(x)
        if self.params['norm_config']=='ln_after':
            x = self.ln(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config, params):
        super().__init__()
        self.params = params.copy()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP_Vit(config, self.params)
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

    def __init__(self, config, params):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        self.params = params.copy()
        for _ in range(config["num_hidden_layers"]):
            block = Block(config, self.params)
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
        self.encoder = Encoder(config, self.params)
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

            


#ResNet101 from https://github.com/mbk2103/ResNet101-Implementation/blob/main/resnet_model.py
# Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self,params, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.params = params.copy()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to handle different input/output dimensions
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        if self.params['norm_config'] == 'bn_before':
            out = self.bn1(out)
        out = self.relu(out)
        if self.params['norm_config'] == 'bn_after':
            out = self.bn1(out)

        out = self.conv2(out)
        if self.params['norm_config'] == 'bn_before':
            out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)
        if self.params['norm_config'] == 'bn_after':
            out = self.bn2(out)

        return out
  
# Define ResNet101v2 model
class ResNet101v2(nn.Module):
    def __init__(self, params, num_classes=1000 ):
        super(ResNet101v2, self).__init__()
        self.params = params.copy()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=23, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(self.params, in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.params, out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.params['norm_config'] == 'bn_before':
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.params['norm_config'] == 'bn_after':
            x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x










#############################################
# 6. Single simulation run (training + evaluation)
#############################################
#############################################
# 6. Single simulation run (training + evaluation)
#############################################
def run_simulation(sim_log_dir, device, sample_index, param_config):
    """
    Run a single simulation experiment.
    All log files are written to sim_log_dir.
    Wandb is initialized for this experiment.
    
    The simulation parameters (learning_rate, batch_size, num_hidden_layers, dataset, offset_value)
    are passed in via the param_config dictionary.
    """
    # --- Extract dataset configuration ---
    dataset_name = param_config.get("dataset", "Gaussian").lower()
    offset_value = param_config.get("offset_value", 0.0)
    
    # --- Load dataset and determine input dimension ---
    # get_dataset_and_input_dim should return a dataset and the input_dim.
    # For Gaussian, it returns a TensorDataset; for MNIST/CIFAR10, a torchvision dataset.
    train_dataset, input_dim = get_dataset_and_input_dim(param_config, device, train=True)
    test_dataset, _ = get_dataset_and_input_dim(param_config, device, train=False)
    
    batch_size = param_config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # --- Compute dataset-specific metrics and prepare data for filtering ---
    if dataset_name == "gaussian":
        # For Gaussian blobs, use the provided parameters.
        dim = input_dim  # e.g., 1000
        center_val = 1.0 / np.sqrt(dim)
        sigma2 = 1.0
        center_positive = np.full((dim,), center_val)
        center_negative = -np.full((dim,), center_val)
        L2_distance = np.linalg.norm(center_positive - center_negative)
        std_val = np.sqrt(sigma2)
        BlobsSeparation = L2_distance / std_val
        print(f"Blobs Separation (normalized): {BlobsSeparation}")
        # Extract training tensor directly from the TensorDataset
        train_X, _ = train_dataset.tensors
    else:
        # For MNIST/CIFAR10, we do not compute BlobsSeparation.
        BlobsSeparation = None
        print(f"Dataset: {param_config['dataset']}")
        # For filtering check, stack a subset of samples (e.g., first 1000) into a tensor.
        subset_size = min(1000, len(train_dataset))
        train_X = torch.stack([train_dataset[i][0] for i in range(subset_size)]).to(device)
    
    # --- Model parameters ---
    num_hidden_layers = param_config["num_hidden_layers"]
    hidden_dim = 100

    train_classes = per_class_counting(train_dataset)

    num_classes = len(train_classes)



    if dataset_name == "gaussian":
        output_dim = 2
    elif param_config.get("class_mapping", None) is not None:
        # Use the number of unique new labels
        output_dim = len(set(param_config["class_mapping"].values()))
    else:
        output_dim = num_classes  # default for full MNIST or CIFAR10

    # --- Wandb initialization ---
    norm_config = param_config.get("norm_config")
    learning_rate = param_config["learning_rate"]
    filtering_mode = param_config["filtering_mode"]  # 'high_igb', 'low_igb', or 'none'
    
    # Construct group name and tags including dataset and offset.
    group_name = (f"dataset_{param_config['dataset']}_offset_{offset_value}_"
                  f"NormMode_{norm_config}_depth_{num_hidden_layers}_"
                  f"lr_{learning_rate}_Bs_{batch_size}_Filtering_{filtering_mode}")
    run_name = f"Sample{sample_index}"
    wandb_id = wandb.util.generate_id()
    tags = [
        f"dataset_{param_config['dataset']}",
        f"offset_{offset_value}",
        f"LR_{learning_rate}",
        f"BS_{batch_size}",
        f"NormMode_{norm_config}",
        f"Depth_{num_hidden_layers}",
        f"Filtering_{filtering_mode}",
        f"Model_{param_config.get('model', 'MLP')}"
    ]
    # Optionally, include BlobsSeparation tag if available.
    if BlobsSeparation is not None:
        tags.append(f"BlobsSeparation_{BlobsSeparation:.2f}")

    run = wandb.init(project= 'MNIST_DEBUG_New', #'MLP_exp_RealData_MNIST_Final',
                     group=group_name,
                     name=run_name,
                     id=wandb_id,
                     resume="allow",
                     reinit=True,
                     tags=tags,
                     notes="Experiments to compare the effect of IGB on MLP with a balanced dataset",
                     entity="emanuele_francazi")
    
    # Customize the default x-axis for logged metrics
    CustomizedX_Axis()
    
    wandb.config.update({
        "learning_rate": learning_rate,
        "epochs": 200,
        "batch_size": batch_size,
        "norm_config": norm_config,
        "num_hidden_layers": num_hidden_layers,
        "dataset": param_config["dataset"],
        "offset_value": offset_value,
        "model": model_name
    })
    
    # --- Clear log files in sim_log_dir ---
    files_to_clear = [
        'eval_times.txt',
        'train_global_loss.txt', 'train_global_accuracy.txt', 'train_frac0.txt', 'train_frac1.txt', 'train_max_frac.txt',
        'train_class0_loss.txt', 'train_class0_accuracy.txt', 'train_class1_loss.txt', 'train_class1_accuracy.txt',
        'test_global_loss.txt', 'test_global_accuracy.txt', 'test_frac0.txt', 'test_frac1.txt', 'test_max_frac.txt',
        'test_class0_loss.txt', 'test_class0_accuracy.txt', 'test_class1_loss.txt', 'test_class1_accuracy.txt'
    ]
    for f in files_to_clear:
        open(os.path.join(sim_log_dir, f), 'w').close()
    
    # --- Model creation ---
    if model_name == 'MLP':
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers,
                    output_dim=output_dim, norm_config=norm_config)
    elif model_name in ['Swin_T','Swin_S','Swin_B','Swin_L']:
        if dataset_name == 'tinyimagenet':
            input_resolution = 64
        elif dataset_name == 'cifar10':
            input_resolution = 32
        elif dataset_name == 'mnist':
            input_resolution = 28
        else:
            input_resolution = 224  # fallback

        if norm_config not in ('ln_before','ln_after'):
            raise ValueError("For Swin models, use norm_config 'ln_before' or 'ln_after'.")

        ctor = {'Swin_T': Swin_T, 'Swin_S': Swin_S, 'Swin_B': Swin_B, 'Swin_L': Swin_L}[model_name]
        model = ctor(
            num_classes=output_dim,
            input_resolution=input_resolution,
            drop_path_rate=0.2,
            norm_config=norm_config
        )

    elif param_config['model'] == 'ViT':

        # Auto-get image size/channels for config
        sample_img, _ = train_dataset[0]
        input_channels = sample_img.shape[0]
        input_image_size = sample_img.shape[1]  # assumes square; use shape[1], shape[2] if not


        config = {
    "patch_size": 4,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": input_image_size,    # <-- Automatically set!
    "num_classes": num_classes,
    "num_channels": input_channels,    # <-- Automatically set!
    "qkv_bias": False,
}
        model = ViTForClassfication(config, param_config)


    elif param_config['model'] == 'ResNet101':
        model = ResNet101v2(param_config, num_classes=output_dim)
    else:
        raise ValueError(f"Unknown model {model_name}")

    model.to(device)
    # --- Filtering mode (if used) ---
    # Define threshold values as before.
    threshold_map = {
        'low_igb': 0.1,
        'high_igb': 0.9,
        'none': None
    }
    threshold = threshold_map.get(filtering_mode, None)
    print(f"Filtering mode: {filtering_mode}, Threshold: {threshold}")
    max_attempts = 1000

    if filtering_mode.lower() == 'high_igb':
        counter = 0
        while True:
            counter += 1
            if counter > max_attempts:
                print(f"[Filtering mode High IGB] Maximum attempts reached ({max_attempts}). Exiting simulation.")
                wandb.finish()
                return
            diff, frac0, frac1, class_fractions, max_fraction = filtering_check(model, train_X, device)
            if diff > threshold:
                print(f"[Filtering mode High IGB] Condition met after {counter} iterations: diff = {diff:.4f}")
                break
            else:
                model.init_weights()
    elif filtering_mode.lower() == 'low_igb':
        counter = 0
        while True:
            counter += 1
            if counter > max_attempts:
                print(f"[Filtering mode Low IGB] Maximum attempts reached ({max_attempts}). Exiting simulation.")
                wandb.finish()
                return
            diff, frac0, frac1, class_fractions, max_fraction = filtering_check(model, train_X, device)
            if diff < threshold:
                print(f"[Filtering mode Low IGB] Condition met after {counter} iterations: diff = {diff:.4f}")
                break
            else:
                model.init_weights()
    else:
        print("[Filtering mode] No filtering is performed.")
    
    # --- Ordering Output Nodes ---
    OrderingClassesFlag = 'ON'
    if OrderingClassesFlag == 'ON':
        diff, frac0, frac1, class_fractions, max_fraction = filtering_check(model, train_X, device)
        print(f"Initial fractions: class0 = {frac0:.4f}, class1 = {frac1:.4f}")
        # For binary classification, simply rank the two classes:
        if frac0 >= frac1:
            ordered_mapping = {0: 0, 1: 1}  # class0 is majority
        else:
            ordered_mapping = {0: 1, 1: 0}  # class1 is majority, so we treat it as "ordered class 0"
        print(f"Ordered mapping (ordered index -> original label): {ordered_mapping}")
    else:
        ordered_mapping = {0: 0, 1: 1}
    # --- Training setup ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_train = nn.CrossEntropyLoss()
    criterion_eval  = nn.CrossEntropyLoss(reduction='none')
    
    num_epochs = 80
    num_eval_points = 15
    total_steps = num_epochs * len(train_loader)
    if total_steps > 8:
        eval_steps_log = np.unique(np.logspace(np.log10(9), np.log10(total_steps), num=num_eval_points - 8, dtype=int))
        eval_steps = np.concatenate((np.arange(8), eval_steps_log))
    else:
        eval_steps = np.arange(total_steps)
    eval_steps = np.unique(eval_steps).tolist()
    print("Evaluation will occur at steps:", eval_steps)
    
    # Compute initial normalized weights
    w0 = get_normalized_parameters(model).detach().cpu()

    step_counter = 0
    next_eval_idx = 0
    for epoch in range(num_epochs):
        for batch in train_loader:
            if next_eval_idx < len(eval_steps) and step_counter >= eval_steps[next_eval_idx]:
                train_metrics = evaluate_dataset(model, train_dataset, criterion_eval, device,
                                                 set_type='train', eval_batch_size=128)
                test_metrics  = evaluate_dataset(model, test_dataset, criterion_eval, device,
                                                 set_type='test', eval_batch_size=128)
                ordered_train_metrics = {}
                ordered_test_metrics = {}
                for new_idx, orig_label in ordered_mapping.items():
                    ordered_train_metrics[new_idx] = train_metrics[orig_label]
                    ordered_test_metrics[new_idx] = test_metrics[orig_label] 

                wt = get_normalized_parameters(model).detach().cpu()
                dot_product = torch.dot(w0, wt).item()
                print(f"Step {step_counter}: w0 · wt = {dot_product:.4f}")
                with open(os.path.join(sim_log_dir, "w0_wt_dot.txt"), "a") as f:
                    f.write(f"{step_counter} {dot_product}\n")
                print(f"Step {step_counter}: Train loss={train_metrics['global']['loss']:.4f}, " +
                      f"Train acc={train_metrics['global']['accuracy']:.4f} | " +
                      f"Test loss={test_metrics['global']['loss']:.4f}, Test acc={test_metrics['global']['accuracy']:.4f}")
                # Log raw metrics.
                log_metrics(sim_log_dir, step_counter, train_metrics, test_metrics)
                # Log ordered metrics.
                log_ordered_metrics(sim_log_dir, step_counter, ordered_train_metrics, ordered_test_metrics)
                # Log to wandb.
                wandb.log({
                    'Performance_measures/Train_Accuracy': train_metrics['global']['accuracy'],
                    'Performance_measures/Train_Loss': train_metrics['global']['loss'],
                    'Performance_measures/Train_f0': train_metrics['global']['frac0'],
                    'Performance_measures/Train_max_f': max(train_metrics['global']['frac0'], train_metrics['global']['frac1']),
                    'Performance_measures/Test_Accuracy': test_metrics['global']['accuracy'],
                    'Performance_measures/Test_Loss': test_metrics['global']['loss'],
                    'Performance_measures/Test_f0': test_metrics['global']['frac0'],
                    'Performance_measures/Test_max_f': max(test_metrics['global']['frac0'], test_metrics['global']['frac1']),
                    'Performance_measures/Train_Loss_Class_0': train_metrics[0]['loss'],
                    'Performance_measures/Train_Accuracy_Class_0': train_metrics[0]['accuracy'],
                    'Performance_measures/Train_Loss_Class_1': train_metrics[1]['loss'],
                    'Performance_measures/Train_Accuracy_Class_1': train_metrics[1]['accuracy'],
                    'Performance_measures/Test_Loss_Class_0': test_metrics[0]['loss'],
                    'Performance_measures/Test_Accuracy_Class_0': test_metrics[0]['accuracy'],
                    'Performance_measures/Test_Loss_Class_1': test_metrics[1]['loss'],
                    'Performance_measures/Test_Accuracy_Class_1': test_metrics[1]['accuracy'],
                    # Ordered metrics:
                    'Performance_measures/Train_Accuracy_Ordered_Class_0': ordered_train_metrics[0]['accuracy'],
                    'Performance_measures/Train_Loss_Ordered_Class_0': ordered_train_metrics[0]['loss'],
                    'Performance_measures/Train_Accuracy_Ordered_Class_1': ordered_train_metrics[1]['accuracy'],
                    'Performance_measures/Train_Loss_Ordered_Class_1': ordered_train_metrics[1]['loss'],
                    'Performance_measures/Test_Accuracy_Ordered_Class_0': ordered_test_metrics[0]['accuracy'],
                    'Performance_measures/Test_Loss_Ordered_Class_0': ordered_test_metrics[0]['loss'],
                    'Performance_measures/Test_Accuracy_Ordered_Class_1': ordered_test_metrics[1]['accuracy'],
                    'Performance_measures/Test_Loss_Ordered_Class_1': ordered_test_metrics[1]['loss'],

                    'w0_wt_dot': dot_product,
                    'Performance_measures/True_Steps_+_1': step_counter + 1,
                })
                next_eval_idx += 1

            model.train()
            optimizer.zero_grad()
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion_train(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            step_counter += 1

    print("Training completed for this simulation.")
    wandb.finish()


def run_init_statistics(combo_log_dir, device, param_config, n_experiments=3000):
    """
    Run multiple independent initializations and log the initial frac0 values.
    This version supports different datasets (Gaussian, MNIST, CIFAR10) with an added offset.
    
    The param_config dictionary must include:
      - "dataset": one of "Gaussian", "MNIST", "CIFAR10"
      - "offset_value": a float value to add to the standardized images (for MNIST/CIFAR10)
      - Other parameters like "num_hidden_layers" and "norm_config".
    
    All frac0 values for a given configuration are appended to a single file in combo_log_dir.
    """
    # --- Load the training dataset and determine the input dimension ---
    # get_dataset_and_input_dim should return (dataset, input_dim)
    train_dataset, input_dim = get_dataset_and_input_dim(param_config, device, train=True)
    dataset_name = param_config.get("dataset", "Gaussian").lower()
    model_name = param_config.get('model', 'MLP')
    
    # --- Prepare training data for filtering ---
    if dataset_name == "gaussian":
        # For Gaussian, the dataset is a TensorDataset.
        train_X, _ = train_dataset.tensors
    else:
        # For MNIST/CIFAR10, extract a subset of images and stack them into a single tensor.
        subset_size = min(1000, len(train_dataset))
        train_X = torch.stack([train_dataset[i][0] for i in range(subset_size)]).to(device)
    
    # --- Model parameters ---
    hidden_dim = 100

    train_classes = per_class_counting(train_dataset)

    num_classes = len(train_classes)


    if dataset_name == "gaussian":
        output_dim = 2
    elif param_config.get("class_mapping", None) is not None:
        # Use the number of unique new labels
        output_dim = len(set(param_config["class_mapping"].values()))
    else:
        output_dim = num_classes  # default for full MNIST or CIFAR10


    # Define file path for storing the frac0 values for this configuration
    log_file_path = os.path.join(combo_log_dir, 'init_frac0.txt')
    # Define file paths for the new logs
    log_file_maxf = os.path.join(combo_log_dir, 'init_max_f.txt')
    log_file_vector = os.path.join(combo_log_dir, 'init_frac_vector.txt')
    
    # For each independent initialization:
    for experiment in range(1, n_experiments + 1):
        # Create the model using the given param_config.
        if model_name == 'MLP':
            model = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
                        num_hidden_layers=param_config["num_hidden_layers"],
                        output_dim=output_dim, norm_config=param_config["norm_config"])
        elif model_name in ['Swin_T','Swin_S','Swin_B','Swin_L']:
            if dataset_name == 'tinyimagenet':
                input_resolution = 64
            elif dataset_name == 'cifar10':
                input_resolution = 32
            elif dataset_name == 'mnist':
                input_resolution = 28
            else:
                input_resolution = 224
            norm_config = param_config["norm_config"]
            if norm_config not in ('ln_before','ln_after'):
                raise ValueError("For Swin models, use norm_config 'ln_before' or 'ln_after'.")
            ctor = {'Swin_T': Swin_T, 'Swin_S': Swin_S, 'Swin_B': Swin_B, 'Swin_L': Swin_L}[model_name]
            model = ctor(num_classes=output_dim, input_resolution=input_resolution, drop_path_rate=0.2, norm_config=norm_config)


        elif param_config['model'] == 'ViT':

            # Auto-get image size/channels for config
            sample_img, _ = train_dataset[0]
            input_channels = sample_img.shape[0]
            input_image_size = sample_img.shape[1]  # assumes square; use shape[1], shape[2] if not

            config = {
        "patch_size": 4,
        "hidden_size": 48,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 4 * 48,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "image_size": input_image_size,    # <-- Automatically set!
        "num_classes": num_classes,
        "num_channels": input_channels,    # <-- Automatically set!
        "qkv_bias": False,
    }
            model = ViTForClassfication(config, param_config)


        elif param_config['model'] == 'ResNet101':
            model = ResNet101v2(param_config, num_classes=output_dim)
        else:
            raise ValueError(f"Unknown model {model_name}")
        model.to(device)
        
        # Compute initial fractions without any reinitialization loop
        diff, frac0, frac1, class_fractions, max_fraction = filtering_check(model, train_X, device)
        #print(f"Experiment {experiment} for config {param_config}: Initial frac0 = {frac0:.4f}, frac1 = {frac1:.4f}")
        # Optionally perform ordering if desired
        OrderingClassesFlag = 'OFF'  # Alternatively, this flag could be set via param_config
        if OrderingClassesFlag == 'ON':
            diff, frac0, frac1, class_fractions, max_fraction = filtering_check(model, train_X, device)
            #print(f"Initial fractions: class0 = {frac0:.4f}, class1 = {frac1:.4f}")
            # For binary classification, simply rank the two classes:
            if frac0 >= frac1:
                # Append the frac0 value to the common log file for this configuration
                with open(log_file_path, 'a') as f:
                    f.write(f"{frac0}\n")
            else:
                # Append the frac0 value to the common log file for this configuration
                with open(log_file_path, 'a') as f:
                    f.write(f"{frac1}\n")
        else:
            ordered_mapping = {0: 0, 1: 1}
            # Append the frac0 value to the common log file for this configuration
            with open(log_file_path, 'a') as f:
                f.write(f"{frac0}\n")

            # Append the max fraction to its log file
            with open(log_file_maxf, 'a') as f:
                f.write(f"{max_fraction}\n")

            # Append the full vector of class fractions (space-separated) to its log file
            with open(log_file_vector, 'a') as f:
                f.write(" ".join([f"{frac:.8f}" for frac in class_fractions]) + "\n")



#############################################
# 7. Main: Outer loop over simulation experiments with a parameter grid
#############################################
def main():
    device_str = 'cuda:1'  #'cuda:0'  # or 'cpu'
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    ModelFlag = 'ResNet101'  #'MLP'  # or 'ViT'  # or 'ResNet101'
    
    # Set the RunMode flag:
    RunMode = 'InitStatistics'  #'Dynamics'  # or 'InitStatistics'

    # Define a parameter grid for simulations.
    # To add/change parameters, simply modify this dictionary.

    param_grid = {
    'dataset': ['tinyimagenet'], # 'Gaussian', 'MNIST', 'CIFAR10'
    'offset_value': [0.0],
    'learning_rate': [0.00001],
    'batch_size': [512],
    'num_hidden_layers': [1],#[1, 20],
    'norm_config': ['bn_before','bn_after'], #['bn_before','bn_after'], #['none', 'ln_before','ln_after'], #['bn_before'], #['none', 'bn_before', 'ln_before', 'bn_after', 'ln_after'], #['ln_after'], #['none', 'bn_before', 'ln_before', 'bn_after', 'ln_after'], # 'none', 'bn_before', 'ln_before', 'bn_after', 'ln_after'
    'filtering_mode': ['none'],  # 'high_igb', 'low_igb', or 'none'
    'class_mapping': [None], #[{0:0, 1:0, 3:1, 4:1, 5:1, 7:1, 8:0, 9:0}], #[{0:0, 1:1, 2:0, 3:1, 4:0, 5:1, 6:0, 7:1, 8:0, 9:1}], #[{0:0, 1:0, 3:1, 4:1, 5:1, 7:1, 8:0, 9:0}]#[{3:0, 5:1}]#[{3:0, 5:1}]# [{0:0, 1:1, 2:0, 3:1, 4:0, 5:1, 6:0, 7:1, 8:0, 9:1}] #if you set "class_mapping": None the dataset is not filtered. Otherwise, the mapping dictionary is used for filtering/aggregation.
    'n_per_class': ['min'],  # Number of samples per class to select (None for all); can be a None dict or int or 'min'
    'model': [ModelFlag],  # 'MLP', 'ViT', or 'ResNet101'
}
    
    # Use itertools.product to generate all parameter combinations.
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))

    n_experiments = 10000 #10#5000  # number of independent runs (samples)

    if RunMode == 'Dynamics':
        base_log_dir = './logs'
        if not os.path.exists(base_log_dir):
            os.makedirs(base_log_dir)
        # For each parameter combination, create a subfolder and run n_experiments per combination.
        for sample_index in range(1, n_experiments + 1):
            print(f"Starting simulation Sample {sample_index} for all parameter combinations...")
            for combo in combinations:
                # Create a dictionary for the current parameter combination.
                param_config = dict(zip(keys, combo))
                # Create a folder name that encodes the parameter values, including dataset and offset.
                combo_folder = (
                    f"dataset_{param_config['dataset']}_offset_{param_config['offset_value']}_"
                    f"lr_{param_config['learning_rate']}_Bs_{param_config['batch_size']}_"
                    f"depth_{param_config['num_hidden_layers']}_norm_{param_config['norm_config']}_"
                    f"Filt_{param_config['filtering_mode']}"
                )
                combo_log_dir = os.path.join(base_log_dir, combo_folder)
                if not os.path.exists(combo_log_dir):
                    os.makedirs(combo_log_dir)
                
                sim_log_dir = os.path.join(combo_log_dir, f"Sample{sample_index}")
                if not os.path.exists(sim_log_dir):
                    os.makedirs(sim_log_dir)
                print(f"  Running simulation for parameter combination {param_config} ...")
                run_simulation(sim_log_dir, device, sample_index, param_config)
                print(f"  Simulation for parameter combination {param_config} completed.\n")
            print(f"Completed all parameter combinations for simulation Sample {sample_index}\n")
    elif RunMode == 'InitStatistics':
        base_log_dir = './logs/InitStatistics'
        for combo in combinations:
            param_config = dict(zip(keys, combo))
            # Create one folder per configuration that encodes dataset and offset
            combo_folder = (
                f"dataset_{param_config['dataset']}_offset_{param_config['offset_value']}_"
                f"lr_{param_config['learning_rate']}_Bs_{param_config['batch_size']}_"
                f"depth_{param_config['num_hidden_layers']}_norm_{param_config['norm_config']}_"
                f"Filt_{param_config['filtering_mode']}"
            )
            combo_log_dir = os.path.join(base_log_dir, combo_folder)
            if not os.path.exists(combo_log_dir):
                os.makedirs(combo_log_dir)
            # Call the new function to run n_experiments for this configuration
            run_init_statistics(combo_log_dir, device, param_config, n_experiments=n_experiments)
    else:
        print("Invalid RunMode specified!")

if __name__ == '__main__':
    main()

































