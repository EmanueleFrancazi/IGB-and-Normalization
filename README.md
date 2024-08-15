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
  - This subfolder contains an adaptation of the [ViT-pytorch repository](https://github.com/jeonsworld/ViT-pytorch), which is a PyTorch reimplementation of Google's [Vision Transformer (ViT) repository](https://github.com/google-research/vision_transformer).
  - This serves as an example of how the code for a specific project can be re-adapted to investigate IGB.

### `utils/`

This folder contains functions necessary to integrate the experiments and investigations on IGB. The utilities provided here must be imported into the codes in the `RunsCode` folder.

### `requirements.txt`

The repository includes a `requirements.txt` file located in the root directory. This file lists all the Python packages required to run the codes in this repository. To install these dependencies, run the following command from the project root:

```bash
pip install -r requirements.txt
```

## Running the Code

To ensure proper functionality, all runs should be initiated from the project root directory. This setup guarantees that the absolute imports will work correctly, allowing the `MultiModels` scripts to access the necessary modules from the `utils` folder.

---

For further details on specific functionalities or to contribute, please refer to the respective scripts and documentation within each folder.

