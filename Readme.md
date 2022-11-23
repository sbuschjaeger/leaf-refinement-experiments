# Improving the Accuracy-Memory Trade-Off of Random Forests Via Leaf-Refinement

This is the repository for the paper "Improving the Accuracy-Memory Trade-Off of Random Forests Via Leaf-Refinement". The repository is structured as the following:

- `PyPruning`: This repository contains the implementations for all pruning algorithms and can be installed as a regular python package and used in other projects. For more information have a look at the Readme file in `PyPruning/Readme.md` and its documentation in `PyPruning/docs`.
- `experiment_runner`: This is a simple package / script which can be used to run multiple experiments in parallel on the same machine or distributed across many different machines. It can also be installed as a regular python package and used for other projects. For more information have a look at the Readme file in `experiment_runner/Readme.md`.
- `{adult, bank, connect, ..., wine-quality}`: Each folder contains an script `init.sh` which downloads the necessary files and performs pre-processing if necessary (e.g. extract archives etc.). 
- `init_all.sh`: Iterates over all datasets and calls the respective `init.sh` files. Depending on your internet connection this may take some time
- `environment.yml`: Anaconda environment file which contains all dependencies. For more details see below
- `LeafRefinement.py`: This is the implementation of the LeafRefinement method. We initially implemented a more complex method which uses Proximal Gradient Descent to simultaneously learn the weights and refine leaf nodes. During our experiments we discovered that leaf-refinement in iteself was enough and much simpler. We kept our old code, but implemented the `LeafRefinement.py` class for easier usage.
- `run.py`: The script which executes the experiments. For more details see the examples below.
- `plot_results.py`: The script is used explore and display results. It also creates the plots for the paper.

## Getting everything ready

This git repository contains two submodules `PyPruning` and `experiment_runner` which need to be cloned first. 

    git clone --recurse-submodules git@github.com:sbuschjaeger/leaf-refinement-experiments.git

After the code has been obtained you need to install all dependencies. If you use `Anaconda` you can simply call

    conda env create -f environment.yml

to prepare and activate the environment `LR`. After that you can install the python packages `PyPruning` and `experiment_runner` via pip:

    pip install -e file:PyPruning
    pip install -e file:experiment_runner

and finally activate the environment with

    conda activate LR

Last you will need to get some data. If you are interested in a specific dataset you can use the accompanying `init.sh` script via

    cd `${Dataset}`
    ./init.sh

or if you want to download all datasets use

    ./init_all.sh

Depending on your internet connection this may take some time.

## Running experiments

If everything worked as expected you should now be able to run the `run.py` script to prune some ensembles. This script has a decent amount of parameters. See further below for an minimal working example.

- `n_jobs`: Number of jobs / threads used for multiprocessing
- `base`: Base learner used for experiments. Can be {RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, HeterogenousForest}. Can be a list of arguments for multiple experiments. 
- `nl`: Maximum number of leaf nodes (corresponds to scikit-learns max_leaf_nodes parameter)
- `dataset`: Dataset used for experiment. Can be a list of arguments for multiple experiments. 
- `n_estimators`: Number of estimators trained for the base learner.
- `n_prune`: Size of the pruned ensemble. Can be a list of arguments for multiple experiments. 
- `xval`: Number of cross validation runs (default is 5)
- `use_prune`: If set then the script uses a train / prune / test split. If not set then the training data is also used for pruning.
- `timeout`: Maximum number of seconds per run. If the runtime exceeds the provided value, stop execution (default is 5400 seconds)

Note that _all_ base ensembles for all cross validation splits of a dataset are trained before any of the pruning algorithms are used. If you want to evaluate many datasets / hyperparameter configuration in one run this requires a lot of memory. 

To train and prune forests on the `magic` dataset you can for example do

    ./run.py --dataset adult -n_estimators 256 --n_prune 2 4 8 16 32 64 128 256 --nl 64 128 256 512 1024 --n_jobs 128 --xval 5 --base RandomForestClassifier

The results are stored in `${Dataset}/results/${base}/${use_prune}/${date}/results.jsonl` where `${Dataset}` is the dataset (e.g. `magic`) and `${date}` is the current time and date.

In order to re-produce the experiments form the paper you can call:

    ./run.py --dataset adult anura avila bank chess connect eeg elec fashion gas-drift har ida216 japanese-vowels jm1 magic mnist mozilla nursery postures statlog --n_estimators 2 4 8 16 32 64 128 256 --nl 64 128 256 512 1024 2048 --l1_reg 0.01 0.05 0.1 0.5 1.0 --n_jobs 128 --xval 5 --base RandomForestClassifier

**Important:** This call uses 128 threads and requires a decent (something in the range of 64GB) amount of memory to work. 

## Exploring the results

After you run the experiments you can view the results with the `plot_results.py` script. We recommend to use an interactive Python environment for that such as Jupyter or VSCode with the ability to execute cells, but you should also be able to run this script as-is. This script is fairly well-commented, so please have a look at it for more detailed comments. 