# Algorithms

This repository contains code to reproduce experiments analyzing the effects of normalization layers on IGB (Initial Guessing Bias). The code structure is designed to ensure compatibility with external repositories/projects with minimal edits, allowing for the extension of experiments and the study of IGB across different projects.

**Note:** The codes in the `MultiModels` folder import modules from the `utils` folder. Runs must be launched from the project root directory (where `MultiModels/` and `utils/` are subdirectories) to use absolute imports.

## Repository Structure

The repository is organized as follows:

### `RunsCode/`

This folder contains all the codes used to perform experiments. It includes subfolders, each associated with a specific model (or a group of models) that can be selected for the study of IGB. The default subfolders are:

- **`MultiModels/`**
  - This is the base code, which contains several different architectures that can be employed for experiments. The folder includes a run manager script, where you can set various hyper-parameters for the simulations.
  - To run simulations, navigate to the `Algorithm` folder and execute the following command:

    ```bash
    ./RunsCode/MultiModels/PythonRunManager.sh n1 n2 > code.out 2> code.err &
    ```

    This command will perform the simulation for multiple samples, indexed between `n1` and `n2`.

- **`ViT/`**
  - This subfolder contains an adaptation of the [ViT-pytorch repository](https://github.com/jeonsworld/ViT-pytorch), which is a PyTorch reimplementation of Google's [Vision Transformer (ViT) repository](https://github.com/google-research/vision_transformer). This serves as an example of how the code for a specific project can be re-adapted to investigate IGB.

### `utils/`

This folder contains essential functions and utilities required to integrate and perform experiments on IGB (Initial Guessing Bias). The key script within this folder is `IGB_utils.py`, which includes all the methods necessary for conducting experiments related to IGB. These utilities must be imported into the codes within the `RunsCode` folder to facilitate the experiments and analyses. For more details on how to properly integrate and use these utilities, refer to the [Study of the Learning Dynamics](#2-study-of-the-learning-dynamics) section.

### `requirements.txt`

The repository includes a `requirements.txt` file located in the root directory. This file lists all the Python packages required to run the codes in this repository. To install these dependencies, run the following command from the project root:

```bash
pip install -r requirements.txt
```

## Running the Code

Depending on the type of experiment you want to perform, there are two main approaches:

### 1. Statistics at Initialization

If you want to study the model's state at initialization (e.g., analyzing the distribution of observables such as the fraction of datapoints assigned at initialization to the generic class 0, denoted as <img src="https://latex.codecogs.com/png.latex?p_%7Bf_0%7D%5E%7B%5Cchi%7D%28x%29" style="vertical-align: middle;" alt="equation" />
), you can follow these steps:

1. **Include the Model:**
   - Insert your model directly into `./RunsCode/MultiModels/IGB_Exp.py` under the section `#%% Architecture`.
   - Select your model during the instance creation under the section `#%% MODEL INSTANCE`.

2. **Set TrainMode:**
   - Open `./utils/IGB_utils.py` and set the variable `TrainMode = 'OFF'`. This ensures that the experiment stops after computing the statistics at initialization without entering the training phase.

3. **Run Multiple Simulations:**
   - To reconstruct the statistics of a variable (e.g., to reconstruct its distribution over an ensemble of initializations), you may need to run multiple simulations. Since these simulations only take measures at initialization, they are quick to execute.

   - Example command to run 1000 instances of the experiment on different initializations:

    ```bash
    ./RunsCode/MultiModels/PythonRunManager.sh 1 1000 > code.out 2> code.err &
    ```

   This command will automatically perform 1000 instances of the same experiment on different initializations of the model.

### Bash Script

As mentioned, the simulation is started through a bash script (`PythonRunManager.sh`). Within that script, some parameters are set. Specifically:

* **RS**: ReLU Slope. The slope to assign to the customized ReLU (if present).
* **KS**: Kernel size for the MaxPool layer (if present).
* **LR**: The learning rate that will be used. It can be a single value or a set of values (which will be given one after the other).
* **BS**: The batch size that will be used. It can be a single value or a set of values (which will be given one after another).
* **DS**: Shifting constant used for data standardization.

For each of the above parameters, it is also possible to select more than one value. In this case, `(i2 - i1 + 1)` runs will be performed sequentially for each combination of the chosen parameters. For each run, the simulation is started from the bash script using the command:

```bash
python3 ./RunsCode/MultiModels/IGB_Exp.py $i $FolderName $LR $BS $KS $RS $DS
```

The `IGB_Exp.py` script is thus called.

### Reproducibility and Initialization: Random Seed

Immediately after importing the modules in `IGB_Exp.py`, we proceed to initialize the random seeds. Note that initialization must be performed on all libraries that use pseudo-random number generators (in our case, numpy, random, torch).

The operation of fixing the seed for a given simulation is a delicate process since a wrong choice could create an undesirable correlation between random variables generated in independent simulations.

The following two lines fix the seed:

```python
t = int(time.time() * 1000.0)
seed = ((t & 0xff0000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x0000ff) << 24)
```

Python's `time()` method returns the time as a floating-point number expressed in seconds since the epoch, in UTC. This value is then amplified. Finally, the bit order is reversed to reduce dependence on the least significant bits, further increasing the distance between similar values (more details are provided directly in the code, as a comment, immediately after initialization).

The resulting value is then used as a seed for initialization. The seed is saved within a file and printed out, so that the simulation can be easily reproduced if required.

### Logging on Server

To more easily monitor the runs and their results, the code automatically saves logs of relevant metrics on some servers, which can then be accessed at any time to check the status of the simulation.

Specifically, simulation results will be available in:

* **TensorBoard**: No logging is required for this server. For more information on using TensorBoard, see [How to use TensorBoard with PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html).
* **WandB**: You can access the server by creating a new account or through accounts from other portals (e.g., GitHub, Google). For more details, see [W&B - Getting Started with PyTorch](https://docs.wandb.ai/guides/integrations/pytorch) and [Intro to PyTorch with W&B](https://wandb.ai/site/articles/intro-to-pytorch-with-wandb).

### 2. Study of the Learning Dynamics

If you want to study the learning dynamics (i.e., the behavior of the model during training), you can follow the guidelines provided in the [Integration with External Projects](#integration-with-external-projects) section. This involves:

1. **Model Definition:**
   - Define your model by inheriting from the `ImageClassificationBase` class and setting up the necessary layers and parameters.

2. **Prepare for Training:**
   - Initialize storage variables before starting the training process using `model.StoringVariablesCreation()`.

3. **Training Step:**
   - Use the `training_step` method from `IGB_utils.py` to handle the training process, which includes calculating the loss and performing backpropagation.

4. **Validation Step:**
   - At specified checkpoints during training, use the `evaluate` method to assess the model's performance on validation data. This ensures that you can monitor the learning dynamics over time.

## Integration with External Projects

The tools included in this repository, particularly those in `utils/IGB_utils.py`, can be integrated into any external project/model to study IGB with minimal changes to the existing code. Below are the main steps to integrate this repository into a generic project:

### 1. Placement of Project Code

Place your project code inside the `./RunsCode` directory. For example:

```
/project_root
 ├─ RunsCode/
 │   └─ ProjectX/
 │      └─ train.py
 └─ utils/
     └─ IGB_utils.py
```

### 2. Import `IGB_utils.py`

To use the utilities from `IGB_utils.py`, adjust the `sys.path` to ensure the project root is accessible:

```python
import sys
import os

# Dynamically add the project root (two levels up) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from utils.IGB_utils import *
```

### 3. Model Definition

When defining your DNN models, inherit from the `ImageClassificationBase` class to access attributes and methods useful for IGB simulations:

```python
class SimpleMLP(ImageClassificationBase):
    def __init__(self, params):
        super(SimpleMLP, self).__init__()
        # Define layers based on params
```

### 4. Preparing for Training

Before starting the training, initialize variables for storing metrics:

```python
model = to_device(SimpleMLP(params), device)
model.StoringVariablesCreation()  # Initialize storage variables
```

### 5. Folder Structure for Results

Create a directory to store the results of your runs:

```python
# creating folder to store results

FolderPath = './RunsResults/SimulationResult'
if not os.path.exists(FolderPath):
    os.makedirs(FolderPath, exist_ok=True)  
    
# then we create the specific sample folder
SampleFolderPath = FolderPath + '/Data_Shift' + str(shift_const) + '/Sample' + str(args.SampleIndex)
print('The folder created for the sample has the path: ', SampleFolderPath)
if not os.path.exists(SampleFolderPath):
    os.makedirs(SampleFolderPath, exist_ok=True) 
```

**Note:** `SampleIndex` must be one of the arguments passed when launching the code so that a folder associated with the specific run can be created, and all its measurements can be stored within it.

### 6. Define a Dictionary Wrapping All Useful Parameters

After setting up the folder structure, you should define a dictionary to wrap all the useful parameters, including those imported from `IGB_utils.py`:

```python
# wrapping flags/variables from IGB_utils.py
params = {
    'NormMode': NormMode,
    'hidden_sizes': hidden_sizes,
    'n_outputs': n_outputs,
    'input_size': input_size,
    'NormPos': NormPos,
    'Architecture': Architecture,
    'ks': ks,
    'ReLU_Slope': ReLU_Slope,
    'Loss_function': Loss_function,
    'IGB_Mode': IGB_Mode,
    'train_classes': train_classes,
    'valid_classes': valid_classes,
    'num_data_points': {'Train': num_trdata_points, 'Eval': num_valdata_points},
    'label_list': label_list,
    'epochs': epochs,
    'num_tr_batches': num_tr_batches,
    'GradNormMode': GradNormMode,
    'FolderPath': SampleFolderPath,
}
```


### 7. Training Step

Use the `training_step` method from `IGB_utils.py` to handle training:

```python
for batch in train_loader:
    Res = model.training_step(batch, num_trdata_points, params)
    loss = Res['loss']
    train_losses.append(loss)
    loss.backward()
```

### 8. Validation Step

Perform evaluation using the `evaluate` method at specific points during training:

```python
if (step + 1) in ValChecks:  # Validation phase
    test_result = evaluate(model, val_loader, 'Eval', params)
    train_result = evaluate(model, train_loader, 'Train', params)
    WandB_logs(step + 1, model)  # Log on WandB
    save_on_file(model, params)  # Save metrics to file
```

**Note:** The block above is triggered at specific intervals defined by `ValChecks`. The selection of these intervals is flexible. However, `IGB_utils.py` provides a function to set intervals that are equispaced in logspace. To use this function:

```python
# define the intervals for evaluation
N_ValidSteps = 30  # set the number of intervals for performing measurements during the run
num_tr_batches = len(train_DataLoader)  # number of batches
TimeValSteps = ValidTimes(num_epochs, num_tr_batches, N_ValidSteps)
print('epochs with evaluation: ', TimeValSteps)
```

