# Heterogeneous Ensemble Pruning

This is the repository for the paper "Heterogeneous Ensemble Pruning". The repository is structured as the following:

- `PyPruning`: This repository contains the implementations for all pruning algorithms and can be installed as a regular python package and used in other projects. For more information have a look at the Readme file in `PyPruning/Readme.md` and its documentation in `PyPruning/docs`.
- `experiment_runner`: This is a simple package / script which can be used to run multiple experiments in parallel on the same machine or distributed across many different machines. It can also be installed as a regular python package and used for other projects. For more information have a look at the Readme file in `experiment_runner/Readme.md`.
- `{adult, bank, connect, ..., wine-quality}`: Each folder contains an script `init.sh` which downloads the necessary files and performs pre-processing if necessary (e.g. extract archives etc.). 
- `init_all.sh`: Iterates over all datasets and calls the respective `init.sh` files. Depending on your internet connection this may take some time
- `environment.yml`: Anaconda environment file which contains all dependencies. For more details see below
- `HeterogenousForest.py`: This is the implementation of the Heterogeneous Forest used as base ensemble. This implementation is based on scikit-learns. For more details check the source code.
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
- `height`: Maximum height of the trees. Corresponds to sci-kit learns `max_depth` parameter. Can be a list of arguments for multiple experiments. Important: Values <= 0 are interpreted as `None` (unlimited tree depth)
- `dataset`: Dataset used for experiment. Can be a list of arguments for multiple experiments. 
- `n_estimators`: Number of estimators trained for the base learner.
- `n_prune`: Size of the pruned ensemble. Can be a list of arguments for multiple experiments. 
- `xval`: Number of cross validation runs (default is 5)
- `use_prune`: If set then the script uses a train / prune / test split. If not set then the training data is also used for pruning.
- `timeout`: Maximum number of seconds per run. If the runtime exceeds the provided value, stop execution (default is 5400 seconds)

Note that _all_ base ensembles for all datasets and all cross validation splits are trained before any of the pruning algorithms are used. If you want to evaluate many datasets / hyperparameter configuration in one run this requires a lot of memory. 

To train and prune forests on the `magic` dataset you can for example do

    ./run --n_jobs 8 --base HeterogenousForest --height 4 8 16 --dataset magic --n_estimators 256 --n_prune 32 --xval 5 --use_prune

The results are stored in `${Dataset}/results/${date}/results.jsonl` where `${Dataset}` is the dataset (e.g. `magic`) and `${date}` is the current time and date. **Important:** If you supply multiple datasets (e.g. `--dataset adult bank connect`) then the results are stored in `combined/results/${date}/results.jsonl`.

In order to re-produce the experiments form the paper you can call:

    ./run.py -d adult bank connect covtype dry-beans eeg elec gas-drift japanese-vowels letter magic mnist mozilla nomao pen-digits satimage shuttle spambase thyroid wine-quality --base RandomForestClassifier ExtraTreesClassifier HeterogenousForest --height 4 8 16 -n 256 -T 4 8 16 32 -j 128 --use_prune

**Important:** This call uses 128 threads and requires a decent (something in the range of 64GB) amount of memory to work. 

## Exploring the results

After you run the experiments you can view the results with the `explore_results.py` script. We recommend to use an interactive Python environment for that such as Jupyter or VSCode with the ability to execute cells, but you should also be able to run this script as-is. This script is fairly well-commented, so please have a look at it for more detailed comments. To change the behavior of it you can adapt the following variables:

`dataset`(line 144): The dataset to be plotted, e.g. `magic` or `multi`
`plot` (line 170): If `true` then the plots from the paper are created. If `false` the raw ranks are displayed
`split_hep` (line 171): If `true` then the HEP method is split into HEP and HEP-LR as done in paper for Q2 and Q3
`split_lambda` (line 172): If `true` then the HEP method is split into the various lambda values as done for Q3
`pval` (line 187): The p-value used for the statistical tests (default is `0.05`)
