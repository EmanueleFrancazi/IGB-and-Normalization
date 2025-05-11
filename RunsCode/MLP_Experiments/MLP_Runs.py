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


import random

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



"""
# this version allow to select the number of datapoints for each label: need to debug

def get_dataset_and_input_dim(param_config, device, train=True):
    dataset_name = param_config.get("dataset", "Gaussian").lower()
    offset_value = param_config.get("offset_value", 0.0)
    n_per_class = param_config.get("n_per_class", None)
    
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
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x + offset_value),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        input_dim = 28 * 28
    elif dataset_name == "cifar10":
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x + offset_value),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        input_dim = 32 * 32 * 3
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
# 3. Filtering mode: check initial imbalance
#############################################
def filtering_check(model, data, device):
    """
    Run a forward pass (without gradients) on the dataset and compute the fraction
    of datapoints predicted as class 0 and class 1.
    Returns the absolute difference and the fractions.
    """
    # Note: you might want to set model.eval() externally if needed.
    with torch.no_grad():
        outputs = model(data)
        preds = outputs.argmax(dim=1)
        frac0 = (preds == 0).float().mean().item()
        frac1 = (preds == 1).float().mean().item()
        diff = abs(frac0 - frac1)
    return diff, frac0, frac1

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

    if dataset_name == "gaussian":
        output_dim = 2
    elif param_config.get("class_mapping", None) is not None:
        # Use the number of unique new labels
        output_dim = len(set(param_config["class_mapping"].values()))
    else:
        output_dim = 10  # default for full MNIST or CIFAR10

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
        f"Filtering_{filtering_mode}"
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
        "offset_value": offset_value
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
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers,
                output_dim=output_dim, norm_config=norm_config)
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
            diff, frac0, frac1 = filtering_check(model, train_X, device)
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
            diff, frac0, frac1 = filtering_check(model, train_X, device)
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
        diff, frac0, frac1 = filtering_check(model, train_X, device)
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

    if dataset_name == "gaussian":
        output_dim = 2
    elif param_config.get("class_mapping", None) is not None:
        # Use the number of unique new labels
        output_dim = len(set(param_config["class_mapping"].values()))
    else:
        output_dim = 10  # default for full MNIST or CIFAR10


    # Define file path for storing the frac0 values for this configuration
    log_file_path = os.path.join(combo_log_dir, 'init_frac0.txt')
    
    # For each independent initialization:
    for experiment in range(1, n_experiments + 1):
        # Create the model using the given param_config.
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, 
                    num_hidden_layers=param_config["num_hidden_layers"],
                    output_dim=output_dim, norm_config=param_config["norm_config"])
        model.to(device)
        
        # Compute initial fractions without any reinitialization loop
        diff, frac0, frac1 = filtering_check(model, train_X, device)
        #print(f"Experiment {experiment} for config {param_config}: Initial frac0 = {frac0:.4f}, frac1 = {frac1:.4f}")
        
        # Optionally perform ordering if desired
        OrderingClassesFlag = 'ON'  # Alternatively, this flag could be set via param_config
        if OrderingClassesFlag == 'ON':
            diff, frac0, frac1 = filtering_check(model, train_X, device)
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


#############################################
# 7. Main: Outer loop over simulation experiments with a parameter grid
#############################################
def main():
    device_str = 'cuda:1'  #'cuda:0'  # or 'cpu'
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    # Set the RunMode flag:
    RunMode = 'InitStatistics'  #'Dynamics'  # or 'InitStatistics'

    # Define a parameter grid for simulations.
    # To add/change parameters, simply modify this dictionary.

    param_grid = {
    'dataset': ['CIFAR10'], # 'Gaussian', 'MNIST', 'CIFAR10'
    'offset_value': [0.0],
    'learning_rate': [0.00001],
    'batch_size': [512],
    'num_hidden_layers': [1, 20],
    'norm_config': ['none', 'bn_before', 'ln_before', 'bn_after', 'ln_after'], #['bn_before'], #['none', 'bn_before', 'ln_before', 'bn_after', 'ln_after'], #['ln_after'], #['none', 'bn_before', 'ln_before', 'bn_after', 'ln_after'], # 'none', 'bn_before', 'ln_before', 'bn_after', 'ln_after'
    'filtering_mode': ['none'],  # 'high_igb', 'low_igb', or 'none'
    'class_mapping': [{0:0, 1:0, 3:1, 4:1, 5:1, 7:1, 8:0, 9:0}], #[{0:0, 1:1, 2:0, 3:1, 4:0, 5:1, 6:0, 7:1, 8:0, 9:1}], #[{0:0, 1:0, 3:1, 4:1, 5:1, 7:1, 8:0, 9:0}]#[{3:0, 5:1}]#[{3:0, 5:1}]# [{0:0, 1:1, 2:0, 3:1, 4:0, 5:1, 6:0, 7:1, 8:0, 9:1}] #if you set "class_mapping": None the dataset is not filtered. Otherwise, the mapping dictionary is used for filtering/aggregation.
    'n_per_class': ['min'],  # Number of samples per class to select (None for all); can be a None dict or int or 'min'
}
    
    # Use itertools.product to generate all parameter combinations.
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))

    n_experiments = 5000 #10#5000  # number of independent runs (samples)

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
