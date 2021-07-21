# Setup Instructions

## Initial Steps
This repo depends on having a conda environment with Pytorch and WandB installed.
Make sure an environment containing both packages is activated prior to running.

## Install CUDA extensions
Navigate to the cpp directory and run

```
python setup.py install 
```
This will build the extensions needed for the sparse Resnets.

## Run Training Script
From your terminal navigate to the root of the project directory and run
```
python train/train.py
```
This will start the training script. If the command is not run from the root of 
the project directory, you will face relative import errors.

