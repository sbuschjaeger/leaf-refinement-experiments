# Heterogeneous Ensemble Pruning

This is the repository for the paper "Heterogeneous Ensemble Pruning". The repository is structed as the following:

- `PyPruning`: This repository contains the implementations for all pruning algorithms and can be installed as a regular python package and used in other projects. For more information have a look at the Readme file in `PyPruning/Readme.md` and its documentation in `PyPruning/docs`.
- `experiment_runner`: This is a simple package / script which can be used to run multiple experiments in parallel on the same machine or distributed across many different machines. It can also be installed as a regular python package and used for other projects. For more information have a look at the Readme file in `experiment_runner/Readme.md`.
- `{adult, bank, connect, ..., wine-quality}`: Each folder contains an script `init.sh` which downloads the necessary files and performs pre-processing if necessary (e.g. extract archives etc.). 
- `init_all.sh`: Iterates over all datasets and calls the respetive `init.sh` files. Depending on your internet conection this may take some time
- `environment.yml`: Anaconda environment file which contains all dependencies. For more details see below
- `HeterogenousForest.py`: This is the implementation of the Heterogenous Forest used as base ensemble. This implementation is based on scikit-learns. For more details check the source code.
- `run.py`: The script which executes the experiments. For more details see the examples below.
- `explore_results.py`: The script is used explore and display results. It also creates the plots for the paper.

## Getting everything ready

This git repository contains two submodules `PyPruning` and `experiment_runner` which need to be cloned first. 

    (removed to preserve anonymity)

**Note for reviewers @ ECML-PKDD2020**: For your convenience and to not break the double-blind reviewing process, we already included all submodules in this submission. Additionally, we deleted all git-related files to reduce information leakage of the authors.

After the code has been obtained you need to install all dependencies. If you use `Anaconda` you can simply call

    conda env create -f environment.yml

to prepare and activate the environment `hep`. After that you can install the python packages `PyPruning` and `experiment_runner` via pip:

    pip install -e file:PyPruning
    pip install -e file:experiment_runner

and finally activate the environment with

    conda activate hep

Last you will need to get some data. If you are interested in a specific dataset you can use the accompaniying `init.sh` script via

    cd $Dataset
    ./init.sh

or if you want to download all datasets use

    ./init_all.sh

Depending on your internet connection this may take some time.

## Getting everything ready