import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb  # make sure to install wandb (pip install wandb)
import itertools

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

#############################################
# 6. Single simulation run (training + evaluation)
#############################################
def run_simulation(sim_log_dir, device, sample_index, param_config):
    """
    Run a single simulation experiment.
    All log files are written to sim_log_dir.
    Wandb is initialized for this experiment.
    
    The simulation parameters (learning_rate, batch_size, num_hidden_layers)
    are passed in via the param_config dictionary.
    """
    # === Simulation parameters ===
    dim = 1000                      # number of components
    center_val = 5 #1.0 / np.sqrt(dim) # centers: [1/sqrt(dim), ...] and [-1/sqrt(dim), ...]
    sigma2 = 0.1 #1.0                    # variance
    n_samples_train = 10000
    n_samples_test  = 200

    num_hidden_layers = param_config["num_hidden_layers"]
    hidden_dim = 1000 #100
    output_dim = 2

    # Define the threshold mapping based on filtering mode
    threshold_map = {
        'low_igb': 0.1,
        'high_igb': 0.8,
        'none': None
    }

    # Set filtering mode
    filtering_mode = param_config["filtering_mode"]  # options: 'high_igb', 'low_igb', or 'none'

    # Retrieve the threshold value based on filtering mode
    threshold = threshold_map.get(filtering_mode, None)  # Default to None if filtering_mode is invalid

    print(f"Filtering mode: {filtering_mode}, Threshold: {threshold}")
    max_attempts = 20000


    num_epochs = 15 #80
    num_eval_points = 15

    # === Wandb initialization ===
    #norm_config = param_config.get("norm_config", "bn_after")
    norm_config = param_config.get("norm_config")
    learning_rate = param_config["learning_rate"]
    batch_size = param_config["batch_size"]
    group_name = f'NormMode_{norm_config}_depth_{num_hidden_layers}_lr_{learning_rate}_Bs_{batch_size}_Filtering_{filtering_mode}'
    run_name   = f'Sample{sample_index}'
    wandb_id   = wandb.util.generate_id()
    tags = [f"LR_{learning_rate}", f"BS_{batch_size}", f"NormMode_{norm_config}", f"Depth_{num_hidden_layers}", f"NormMode_{norm_config}", f"Filtering_{filtering_mode}"]
    
    run = wandb.init(project=  'MLP_exp_G_Blobs_FraSetting', #'MLP_exp_G_Blobs_New',
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
      "epochs": num_epochs,
      "batch_size": batch_size,
      "norm_config": norm_config,
      "num_hidden_layers": num_hidden_layers
    })
    
    # === Clear log files in sim_log_dir ===
    files_to_clear = [
        'eval_times.txt',
        'train_global_loss.txt', 'train_global_accuracy.txt', 'train_frac0.txt', 'train_frac1.txt', 'train_max_frac.txt',
        'train_class0_loss.txt', 'train_class0_accuracy.txt', 'train_class1_loss.txt', 'train_class1_accuracy.txt',
        'test_global_loss.txt', 'test_global_accuracy.txt', 'test_frac0.txt', 'test_frac1.txt', 'test_max_frac.txt',
        'test_class0_loss.txt', 'test_class0_accuracy.txt', 'test_class1_loss.txt', 'test_class1_accuracy.txt'
    ]
    for f in files_to_clear:
        open(os.path.join(sim_log_dir, f), 'w').close()
    
    # === Data generation ===
    train_X, train_Y = generate_gaussian_blobs(n_samples_train, dim, center_val, sigma2, device)
    test_X, test_Y   = generate_gaussian_blobs(n_samples_test, dim, center_val, sigma2, device)
    train_dataset = TensorDataset(train_X, train_Y)
    test_dataset  = TensorDataset(test_X, test_Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # === Model creation ===
    model = MLP(input_dim=dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers,
                output_dim=output_dim, norm_config=norm_config)
    model.to(device)

    # === Filtering mode (if used) ===
    if filtering_mode.lower() == 'high_igb':
        counter = 0  # Initialize counter for iterations
        while True:
            counter += 1  # Increment counter at the beginning of each iteration

            # Check if maximum attempts have been reached
            if counter > max_attempts:
                print(f"[Filtering mode High IGB] Maximum attempts reached ({max_attempts}). Exiting simulation.")
                wandb.finish()
                return  # Exit from run_simulation

            diff, frac0, frac1 = filtering_check(model, train_X, device)
            if diff > threshold:
                print(f"[Filtering mode High IGB] Condition met after {counter} iterations: diff = {diff:.4f}")
                break
            else:
                #print(f"[Filtering mode High IGB] Condition NOT met (diff = {diff:.4f}); reinitializing weights. Iteration: {counter}")
                model.init_weights()
    elif filtering_mode.lower() == 'low_igb':
        counter = 0  # Initialize counter for iterations
        while True:
            counter += 1  # Increment counter at the beginning of each iteration

            # Check if maximum attempts have been reached
            if counter > max_attempts:
                print(f"[Filtering mode High IGB] Maximum attempts reached ({max_attempts}). Exiting simulation.")
                wandb.finish()
                return  # Exit from run_simulation

            diff, frac0, frac1 = filtering_check(model, train_X, device)
            if diff < threshold:
                print(f"[Filtering mode Low IGB] Condition met after {counter} iterations: diff = {diff:.4f}")
                break
            else:
                #print(f"[Filtering mode Low IGB] Condition NOT met (diff = {diff:.4f}); reinitializing weights. Iteration: {counter}")
                model.init_weights()

    else:
        print("[Filtering mode] No filtering is performed.")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_train = nn.CrossEntropyLoss()
    criterion_eval  = nn.CrossEntropyLoss(reduction='none')
    
    total_steps = num_epochs * len(train_loader)
    if total_steps > 8:
        eval_steps_log = np.unique(np.logspace(np.log10(9), np.log10(total_steps), num=num_eval_points - 8, dtype=int))
        eval_steps = np.concatenate((np.arange(8), eval_steps_log))
    else:
        eval_steps = np.arange(total_steps)
    eval_steps = np.unique(eval_steps).tolist()
    print("Evaluation will occur at steps:", eval_steps)
    
    step_counter = 0
    next_eval_idx = 0
    for epoch in range(num_epochs):
        for batch in train_loader:

            if next_eval_idx < len(eval_steps) and step_counter >= eval_steps[next_eval_idx]:
                train_metrics = evaluate_dataset(model, train_dataset, criterion_eval, device,
                                                 set_type='train', eval_batch_size=128)
                test_metrics  = evaluate_dataset(model, test_dataset, criterion_eval, device,
                                                 set_type='test', eval_batch_size=128)
                print(f"Step {step_counter}: Train loss={train_metrics['global']['loss']:.4f}, " +
                      f"Train acc={train_metrics['global']['accuracy']:.4f} | " +
                      f"Test loss={test_metrics['global']['loss']:.4f}, Test acc={test_metrics['global']['accuracy']:.4f}")
                # Log to files
                log_metrics(sim_log_dir, step_counter, train_metrics, test_metrics)
                # Log to wandb with per-class metrics included.
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

#############################################
# 7. Main: Outer loop over simulation experiments with a parameter grid
#############################################
def main():
    device_str =  'cuda:1'  #'cuda:0'  # or 'cpu'
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    # Define a parameter grid for simulations.
    # To add/change parameters, simply modify this dictionary.
    param_grid = {
        'learning_rate': [0.00001], #[1e-4, 1e-3, 1e-2, 1e-1], #[1e-3],
        'batch_size': [256],  #[64, 128, 256, 512], #[128], 
        'num_hidden_layers': [1, 20], #[2, 15, 30],
        'norm_config': [ 'bn_before', 'ln_before'], #['bn_after', 'bn_before', 'ln_after', 'ln_before'],  # can be 'bn_before', 'bn_after', 'ln_before', 'ln_after'
        'filtering_mode': ['high_igb', 'low_igb'] # can be 'high_igb', 'low_igb', 'none'
    }
    
    # Use itertools.product to generate all parameter combinations.
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))
    
    base_log_dir = './logs'
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)
    # For each parameter combination, create a subfolder and run n_experiments per combination.    

    n_experiments = 30  # number of independent runs (samples)

    for sample_index in range(1, n_experiments + 1):
        print(f"Starting simulation Sample {sample_index} for all parameter combinations...")
        for combo in combinations:
            # Create a dictionary for the current parameter combination.
            param_config = dict(zip(keys, combo))
            # Create a folder name that encodes the parameter values.
            combo_folder = f"lr_{param_config['learning_rate']}_Bs_{param_config['batch_size']}_depth_{param_config['num_hidden_layers']}_norm_{param_config['norm_config']}_Filt_{param_config['filtering_mode']}"
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


if __name__ == '__main__':
    main()
