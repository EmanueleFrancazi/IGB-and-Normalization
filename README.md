# Algorithms

This repository contains code to reproduce experiments analyzing the effects of normalization layers on IGB (Invariant Gradient-Based). The code structure is designed to ensure compatibility with external repositories/projects with minimal edits, allowing for the extension of experiments and the study of IGB across different projects.

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

This folder contains functions necessary to integrate the experiments and investigations on IGB. The utilities provided here must be imported into the codes in the `RunsCode` folder.

### `requirements.txt`

The repository includes a `requirements.txt` file located in the root directory. This file lists all the Python packages required to run the codes in this repository. To install these dependencies, run the following command from the project root:

```bash
pip install -r requirements.txt
```

## Running the Code

To ensure proper functionality, all runs should be initiated from the project root directory. This setup guarantees that the absolute imports will work correctly, allowing the `MultiModels` scripts to access the necessary modules from the `utils` folder.

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
FolderPath = './RunsResults/' + args.FolderName
if not os.path.exists(FolderPath):
    os.makedirs(FolderPath, exist_ok=True)

# Create specific folder for the sample
SampleFolderPath = f'{FolderPath}/LR{args.learning_rate}/KS{args.ks}/Slope{args.Relu_Slope}/Data_Shift{shift_const}/Sample{args.SampleIndex}'
if not os.path.exists(SampleFolderPath):
    os.makedirs(SampleFolderPath, exist_ok=True)
```

### 6. Training Step
Use the `training_step` method from `IGB_utils.py` to handle training:

```python
for batch in train_loader:
    Res = model.training_step(batch, num_trdata_points, params)
    loss = Res['loss']
    train_losses.append(loss)
    loss.backward()
```

### 7. Validation Step
Perform evaluation using the `evaluate` method during checkpoints:

```python
if (step + 1) in ValChecks:  # Validation phase
    test_result = evaluate(model, val_loader, 'Eval', params)
    train_result = evaluate(model, train_loader, 'Train', params)
    WandB_logs(step + 1, model)  # Log on WandB
    save_on_file(model, params)  # Save metrics to file
```

---

For further details on specific functionalities or to contribute, please refer to the respective scripts and documentation within each folder.

