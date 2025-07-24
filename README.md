# Code-Master-Thesis
 
# Dimension Reduction and Prediction Models

## Overview

This project applies various dimension reduction techniques and prediction models to datasets, with an option to include knowledge-based dimension reduction. 
It allows for scaling, model selection, and hyperparameter tuning through command-line arguments.

## Features

- **Dimension Reduction Techniques**: PCA, Kernel PCA, SVD, Autoencoder
- **Prediction Models**: Logistic Regression, Random Forest, Gradient Boosting Classifier (GBC), Gaussian Naïve Bayes
- **Scaling Methods**: Standard Scaling, Min-Max Scaling
- **Knowledge-Based Dimension Reduction**
- **Hyperparameter Tuning**
- **Multiple Runs Support** for Experiments

## Installation

Ensure you have the required Python packages installed:

pip install numpy pandas scikit-learn argparse


## Usage

Run the script with the following command-line arguments:

python main.py --dataset <DATASET_NAME> --model <MODEL_NAME> [OPTIONS]

### Required Arguments:

- `--dataset <DATASET_NAME>`: Specify the dataset to use.
- `--model <MODEL_NAME>`: Choose a model from: `logistic_regression`, `random_forest`, `gbc`, `gaussian_nb`.

### Optional Arguments:

- `--solver <SOLVER>`: Solver for logistic regression (`lbfgs`, `liblinear`, etc.).
- `--penalty <PENALTY>`: Regularization for logistic regression (`l1`, `l2`).
- `--test_size <FLOAT>`: Proportion of the dataset for testing (default: `0.25`).
- `--random_state <INT>: Set a random seed for reproducibility (default: 42).
- `--dimension_reduction <METHOD>`: Choose from `pca`, `kernel_pca`, `svd`, `auto_encoder`.
- `--number_components <INT>`: Number of components for dimension reduction (default: `1`).
- `--scaling <METHOD>`: Apply scaling (`standard_scaling`, `minmax_scaling`).
- `--kl_based_line`: Apply knowledge-based line reduction.
- `--kl_based_weight`: Apply weighted knowledge-based reduction.
- `--number_runs <INT>`: Number of experiment runs (default: `1`).
- `--number_epochs <INT>`: Number of epochs for training (default: `10`).
- `--number_batch <INT>`: Batch size for training (default: `32`).
- `--parent_weight <FLOAT>`: Minimum parent weight (default: `0.2`).
- `--child_weight <FLOAT>`: Maximum child weight (default: `0.2`).
- `--chosen_level <INT>`: Level for knowledge-based reduction.

## Example Usage

### Running PCA with Logistic Regression:

python main.py --dataset data --model logistic_regression --solver lbfgs --dimension_reduction pca --number_components 5

### Running Random Forest with Standard Scaling:

python main.py --dataset data --model random_forest --scaling standard_scaling

### Running Autoencoder with 50 Epochs:

python main.py --dataset data --model gbc --dimension_reduction auto_encoder --number_epochs 50

## Output

The script prints:

- Training AUC
- Testing AUC
- Testing AUPRC If multiple runs are specified, it also calculates average scores.



