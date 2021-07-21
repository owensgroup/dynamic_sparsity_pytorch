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

## Run the Training Script with Default Parameters
From your terminal navigate to the root of the project directory and run
```
python train/train.py
```
This will start the training script. If the command is not run from the root of 
the project directory, you will face relative import errors.

## Running sweeps with WandB
WandB sweeps are used for hyperparameter tuning and automates the process
of launching and logging multiple runs with slightly different configs. In order
to initiate a sweep, feed in a config file. These are located in the `config`
directory. As an example:
```
wandb sweep config/Resnet18_sweeps.yml
```
This command will return a sweep id (eg. `tejalapeno/ampere_acc_test/2d5pl0gc`). The sweep id is given to the WandB
agent after being generated from the config. This is what you send to srun.
```
srun -p bowser --gpus=1 wandb agent --count=1 tejalapeno/ampere_acc_test/2d5pl0gc
```
I recommend ssh'ing into owens and then using srun to parcel out jobs from each
sweep ID. In time, I'll figure out how to automate this with a bash script.