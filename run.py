#!/usr/bin/env python3

from enum import unique
from operator import index
import time
import pandas as pd
import copy
import numpy as np
import argparse
from multiprocessing import Pool
#from torch.multiprocessing import Pool
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from random import shuffle
import os, psutil
from PyPruning.Papers import create_pruner 

from sklearn.model_selection import StratifiedKFold

from sklearn.utils import parallel_backend

from LeafRefinement import LeafRefinery

from Datasets import get_dataset

# from Metrics import accuracy, avg_accuracy, avg_rademacher, c_bound, n_nodes, n_leaves, effective_height, soft_hinge, mse, bias, diversity

from sklearn.metrics import accuracy_score, f1_score

def accuracy(model, X, target):
    return 100.0*accuracy_score(target, model.predict(X))

def f1(model, X, target):
    return f1_score(target, model.predict(X), average="macro")

def get_n_estimators(model, X, target):
    return len(model.trees) if hasattr(model, "trees") else len(model.estimators_)

def get_n_nodes(model, X, target):
    return sum([e.n_nodes for e in model.trees]) if hasattr(model, "trees") else sum([e.tree_.node_count for e in model.estimators_])

def get_size_kb(model, X, target):
    n_nodes = get_n_nodes(model, X, target)
    return n_nodes * (17 + 4*model.n_classes_ )  / 1024.0

def beautify_scores(scores):
    """Remove test_ in all scores for a nicer output and compute the mean of each score.

    Args:
        scores (dict): A dictionary of the scores in which each score has a list of the corresponding scores, e.g. scores["test_accuracy"] = [0.88, 0.82, 0.87, 0.90, 0.85]

    Returns:
        dict: A dictionary in which all "test_*" keys have been replaced by "*" and all scores are now averaged.  
    """
    nice_scores = {}
    for key, val in scores.items():
        if "test" in key:
            key = key.replace("test_","")

        nice_scores[key] = np.mean(val)
        nice_scores[key + "_std"] = np.std(val)

    return nice_scores

def merge_dictionaries(dicts):
    """Merges the given list of dictionaries into a single dictionary:
    [
        {'a':1,'b':2}, {'a':1,'b':2}, {'a':1,'b':2}
    ]
    is merged into
    {
        'a' : [1,1,1]
        'b' : [2,2,2]
    }

    Args:
        dicts (list of dicts): The list of dictionaries to be merged.

    Returns:
        dict: The merged dictionary.
    """
    merged = {}
    for d in dicts:
        for key, val in d.items():
            if key not in merged:
                merged[key] = [val]
            else:
                merged[key].append(val)
    return merged

def run_eval(cfg):
    """Fits and evalutes the model given the current configuration. The cfg tuple is expected to have the following form
        (model, X, Y, scoring, idx, additional_infos, run_id)
    This function basically extracts the train/test indicies supplied by idx given the current run_id:
        train, test = idx[rid]
        XTrain, YTrain = X[train,:], Y[train]
        XTest, YTest = X[test,:], Y[test]
    and then trains and evaluates a classifier on XTrain/Ytrain and XTest/YTest. Any additional_infos passed to this function are simply returned which makes housekeeping a little easier. 

    Args:
        cfg (tuple): A tuple of the form (model, X, Y, scoring, idx, additional_infos, run_id)

    Returns:
        a tuple of (dict, dict): The first dictionary is the result of the evaluation of the form 
        {
            'test_accuracy': 0.8,
            'train_accuracy': 0.9, 
            # ....
        }. The second dictionary contains the additional_infos passed to this function. 
    """
    model, scoring, additional_infos, base = cfg["model"], cfg["scoring"], cfg["additional_infos"], cfg["base"]
    XTrain, YTrain, XTest, YTest = cfg["X_train"], cfg["Y_train"], cfg["X_test"], cfg["Y_test"]

    # if "idx" in cfg:
    #     X, Y, idx = cfg["X"], cfg["Y"], cfg["idx"]
    #     train, test = idx[rid]
    #     XTrain, YTrain = copy.deepcopy(X[train,:]), copy.deepcopy(Y[train])
    #     XTest, YTest = copy.deepcopy(X[test,:]), copy.deepcopy(Y[test])
    #     base = copy.deepcopy(forest[rid])
    #     print("RUNNING XVAL {}".format(rid))
    # else:
    #     XTrain, YTrain = copy.deepcopy(cfg["X_train"]), copy.deepcopy(cfg["y_train"])
    #     XTest, YTest = copy.deepcopy(cfg["X_test"]), copy.deepcopy(cfg["y_test"])
    #     base = copy.deepcopy(forest)
    
    import os, psutil
    process = psutil.Process(os.getpid())
    # print("BEFORE FIT: {} Mb".format(process.memory_info().rss / 10**6)) 

    start = time.time()
    if model is None:
        model = base
    else:
        model.prune(XTrain, YTrain, base.estimators_, base.classes_, base.n_classes_)
    end = time.time()

    scores = {}
    for name, method in scoring.items():
        scores["test_{}".format(name)] = method(model, XTest, YTest)
        scores["train_{}".format(name)] = method(model, XTrain, YTrain)
    
    # additional_infos["model"] = model

    scores["train_time_sec"] = end - start 
    return scores, additional_infos

def prepare_xval(cfg):
    """Small helper function which copies the given config xval times and inserts the correct run_id for running cross-validations.

    Args:
        cfg (dict): The configuration.

    Returns:
        list: A list of configurations.
    """
    
    cfgs = []
    if "X" in cfg and "Y" in cfg:
        X, Y, forest, idx = cfg["X"], cfg["Y"], cfg["forest"], cfg["idx"]
        del cfg["X"]
        del cfg["Y"]
        del cfg["forest"]
        del cfg["idx"]

        for i, (train, test) in enumerate(idx):
            tmp = copy.deepcopy(cfg)

            tmp["X_train"] = X[train,:]
            tmp["Y_train"] = Y[train]
            tmp["X_test"] = X[test,:]
            tmp["Y_test"] = Y[test]
            tmp["base"] = forest[i]

            cfgs.append(tmp)
    else:
        cfg["base"] = cfg["forest"]
        cfgs.append(cfg)

    return cfgs

def main(args):
    random_state = 42
    np.random.seed(random_state)
    all_df = []
    statistics = []

    if len(args.max_leaves) == 0:
        max_leaves = [None]
    else:
        max_leaves = args.max_leaves

    if len(args.l1_reg) == 0:
        l1_reg = [1e-1]
    else:
        l1_reg = args.l1_reg

    if len(args.n_estimators) == 0:
        n_estimators = [32]
    else:
        n_estimators = args.n_estimators

    max_n_estimators = max(n_estimators)

    n_jobs_in_pool = args.n_jobs

    #client = Client(n_workers=50) #n_workers = args.n_jobs, threads_per_worker = 1, memory_limit = '32GB
    parallel_backend("threading") #, args.n_jobs

    scoring = {
        "accuracy":accuracy,
        "n_estimators":get_n_estimators,
        "size_kb": get_size_kb,
        "n_nodes": get_n_nodes,
        "f1":f1 
    }

    for dataset in args.dataset:
        data = get_dataset(dataset, args.tmpdir)
        if data is None: 
            print("ERROR downloading {}. Skipping".format(dataset))
            continue

        print("Dataset: {}".format(dataset))
        if len(data) == 4:
            X_train,y_train,X_test,y_test = data
            class_dist = Counter(sorted(y_train))
            print("Data: ", X_train.shape, " ", X_train[0:2,:])
            print("Labels: ", y_train.shape, " ", class_dist )
            xval = 1
        else:
            X,Y = data
            print("Data: ", X.shape, " ", X[0:2,:])
            class_dist = Counter(sorted(Y))
            print("Labels: ", Y.shape, " ", class_dist)
            xval = args.xval
            kf = StratifiedKFold(n_splits=args.xval, random_state=random_state, shuffle=True)
            idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X, Y)], dtype=object)

        configs = []
        for nl in max_leaves:
            process = psutil.Process(os.getpid())
            # print("BEFORE LOADING: {} Mb".format(process.memory_info().rss / 10**6)) 

            if len(data) == 4:
                base = RandomForestClassifier(n_estimators=max_n_estimators, max_leaf_nodes=nl, bootstrap = True, random_state=42) 
                base.fit(X_train, y_train)
                common_config = {
                    "X_train":X_train,
                    "Y_train":y_train,
                    "X_test":X_test,
                    "Y_test":y_test,
                    "scoring":scoring,
                    "forest":base
                }
                statistics.append(
                    {
                        "dataset":dataset,
                        "N":X_train.shape[0] + X_test.shape[0],
                        "d":X_train.shape[1],
                        "C":len(set(y_train)), 
                        **{"C_{}".format(k) : v/len(y_train) for k,v in class_dist.items()},
                    }
                )
            else:
                forest = []
                for itrain, _ in idx:
                    rf = RandomForestClassifier(n_estimators=max_n_estimators,  max_leaf_nodes=nl, bootstrap = True, random_state=42) 
                    rf.fit(X[itrain, :], Y[itrain])
                    forest.append(rf)
                
                common_config = {
                    "X": X,
                    "Y": Y, 
                    "scoring":scoring, 
                    "idx":idx,
                    "forest":forest
                }
                statistics.append(
                    {
                        "dataset":dataset,
                        "N":X.shape[0],
                        "d":X.shape[1],
                        "C":len(set(Y)),
                        **{"C_{}".format(k) : v/len(Y) for k,v in class_dist.items()}
                    }
                )
            
            configs.extend(
                prepare_xval(
                    {
                        **common_config,
                        "model": None,
                        "additional_infos": {
                            "dataset":dataset,
                            "LR":False,
                            "pruning":"RF",
                            "max_leafs":nl,
                            "l1":None,
                            "rho":None,
                            "K":None
                        }
                    }
                )
            )

            print("AFTER LOADING: {} Mb".format(process.memory_info().rss / 10**6)) 
            # forest = []
            # for itrain, _ in idx:
            #     rf = RandomForestClassifier(n_estimators=max_n_estimators,  max_leaf_nodes=nl, bootstrap = True, random_state=42) 
            #     rf.fit(X[itrain, :], Y[itrain])
            #     forest.append(rf)
            
            # common_config = {
            #     "X": X,
            #     "Y": Y, 
            #     "scoring":scoring, 
            #     "idx":idx,
            #     "forest":forest
            # }

            for l in l1_reg:
                configs.extend(
                    prepare_xval(
                        {
                            **common_config,
                            "model": LeafRefinery(
                                epochs = 25, 
                                lr = 1e-2, 
                                batch_size = 1024, 
                                optimizer = "adam", 
                                verbose = args.debug,
                                loss_function = "mse", 
                                loss_type = "upper", 
                                l_reg = 1.0,
                                l1_strength = l,
                                pruner = "L1",
                                leaf_refinement=True
                            ),
                            "additional_infos": {
                                "dataset":dataset,
                                "pruning":"L1",
                                "LR":True,
                                "max_leafs":nl,
                                "l1":l,
                                "K":None,
                                "rho":None
                            }
                        }
                    )
                )

                configs.extend(
                    prepare_xval(
                        {
                            **common_config,
                            "model": LeafRefinery(
                                epochs = 25, 
                                lr = 1e-2, 
                                batch_size = 1024, 
                                optimizer = "adam", 
                                verbose = args.debug,
                                loss_function = "mse", 
                                loss_type = "upper", 
                                l_reg = 1.0,
                                l1_strength = l,
                                pruner = "L1",
                                leaf_refinement=False
                            ),
                            "additional_infos": {
                                "dataset":dataset,
                                "pruning":"L1",
                                "LR":False,
                                "max_leafs":nl,
                                "l1":l,
                                "K":None,
                                "rho":None
                            }
                        }
                    )
                )

            for K in n_estimators:
                configs.extend(
                    prepare_xval(
                        {
                            **common_config,
                            "model": LeafRefinery(
                                epochs = 25, 
                                lr = 1e-2, 
                                batch_size = 1024, 
                                optimizer = "adam", 
                                verbose = args.debug,
                                loss_function = "mse", 
                                loss_type = "upper", 
                                l_reg = 1.0,
                                l1_strength = 0,
                                leaf_refinement=True,
                                pruner = create_pruner(method = "random", n_estimators = K)
                            ),
                            "additional_infos": {
                                "dataset":dataset,
                                "pruning":"random",
                                "LR":True,
                                "max_leafs":nl,
                                "l1":None,
                                "rho":None,
                                "K":K
                            }
                        }
                    )
                )

                # for rho in [0.25,0.3,0.35,0.4,0.45,0.5]:
                #     # configs.extend(
                #     #     prepare_xval(
                #     #         {
                #     #             **common_config,
                #     #             "model": LeafRefinery(
                #     #                 epochs = 50, 
                #     #                 lr = 1e-1, 
                #     #                 batch_size = 1024, 
                #     #                 optimizer = "adam", 
                #     #                 verbose = args.debug,
                #     #                 loss_function = "mse", 
                #     #                 loss_type = "upper", 
                #     #                 l_reg = 1.0,
                #     #                 l1_strength = 0,
                #     #                 pruner = create_pruner(method = "drep", n_estimators = K, **{"metric_options":{"rho":rho}})
                #     #             ),
                #     #             "additional_infos": {
                #     #                 "dataset":dataset,
                #     #                 "LR":True,
                #     #                 "pruning":"drep",
                #     #                 "max_leafs":nl,
                #     #                 "l1":None,
                #     #                 "rho":rho,
                #     #                 "K":K
                #     #             }
                #     #         }
                #     #     )
                #     # )

                #     configs.extend(
                #         prepare_xval(
                #             {
                #                 **common_config,
                #                 "model": LeafRefinery(
                #                     epochs = 0, 
                #                     lr = 0, 
                #                     batch_size = 1024, 
                #                     optimizer = "adam", 
                #                     verbose = args.debug,
                #                     loss_function = "mse", 
                #                     loss_type = "upper", 
                #                     l_reg = 1.0,
                #                     l1_strength = 0,
                #                     pruner = create_pruner(method = "drep", n_estimators = K, **{"metric_options":{"rho":rho}})
                #                 ),
                #                 "additional_infos": {
                #                     "dataset":dataset,
                #                     "LR":False,
                #                     "pruning":"drep",
                #                     "max_leafs":nl,
                #                     "l1":None,
                #                     "rho":rho,
                #                     "K":K
                #                 }
                #             }
                #         )
                #     )

                # for pname, poptions in [
                #     ("individual_contribution",{"n_jobs":1}),
                #     ("individual_error",{"n_jobs":1}),
                #     ("reduced_error",{"n_jobs":1}),
                #     ("complementariness",{"n_jobs":1}),
                #     ("random",{}),
                #     ("largest_mean_distance",{"selector_options":{"n_jobs":1}})
                #     ]:
                #     # configs.extend(
                #     #     prepare_xval(
                #     #         {
                #     #             **common_config,
                #     #             "model": LeafRefinery(
                #     #                 epochs = 50, 
                #     #                 lr = 1e-1, 
                #     #                 batch_size = 1024, 
                #     #                 optimizer = "adam", 
                #     #                 verbose = args.debug,
                #     #                 loss_function = "mse", 
                #     #                 loss_type = "upper", 
                #     #                 l_reg = 1.0,
                #     #                 l1_strength = 0,
                #     #                 pruner = create_pruner(method = pname, n_estimators = K, **poptions)
                #     #             ),
                #     #             "additional_infos": {
                #     #                 "dataset":dataset,
                #     #                 "pruning":pname,
                #     #                 "LR":True,
                #     #                 "max_leafs":nl,
                #     #                 "l1":None,
                #     #                 "rho":None,
                #     #                 "K":K
                #     #             }
                #     #         }
                #     #     )
                #     # )

                #     configs.extend(
                #         prepare_xval(
                #             {
                #                 **common_config,
                #                 "model": LeafRefinery(
                #                     epochs = 0, 
                #                     lr = 0, 
                #                     batch_size = 1024, 
                #                     optimizer = "adam", 
                #                     verbose = args.debug,
                #                     loss_function = "mse", 
                #                     loss_type = "upper", 
                #                     l_reg = 1.0,
                #                     l1_strength = 0,
                #                     leaf_refinement=False,
                #                     pruner = create_pruner(method = pname, n_estimators = K, **poptions)
                #                 ),
                #                 "additional_infos": {
                #                     "dataset":dataset,
                #                     "pruning":pname,
                #                     "LR":False,
                #                     "max_leafs":nl,
                #                     "l1":None,
                #                     "rho":None,
                #                     "K":K
                #                 }
                #             }
                #         )
                #     )
                

        # print("BEFORE DEBUG: {} Mb".format(process.memory_info().rss / 10**6)) 

        if args.debug:
            delayed_metrics = []
            for cfg in configs:
                delayed_metrics.append(run_eval(cfg))
        else:
            print("Configured {} experiments. Starting experiments now using {} jobs.".format(len(configs), n_jobs_in_pool))
            pool = Pool(n_jobs_in_pool)
            delayed_metrics = []
            
            shuffle(configs)
            for eval_return in tqdm(pool.imap_unordered(run_eval, configs), total=len(configs)):
                delayed_metrics.append(eval_return)
        
        metrics = []
        names = list(set(["_".join([str(val) for val in cfg.values()]) for _, cfg in delayed_metrics]))
        for n in names:
            tmp = [scores for scores, cfg in delayed_metrics if "_".join([str(val) for val in cfg.values()]) == n]
            cfg = [cfg for _, cfg in delayed_metrics if "_".join([str(val) for val in cfg.values()]) == n][0]

            metrics.append(
                {
                    **beautify_scores(merge_dictionaries(tmp)),
                    **cfg
                }
            )
             
        df = pd.DataFrame(metrics)
        df = df.sort_values(by=["dataset", "n_estimators", "max_leafs", "pruning", "LR", "l1", "rho", "K"])
        df.to_csv("{}_{}.csv".format(dataset, datetime.now().strftime("%d-%m-%Y-%H:%M")),index=False)
        with pd.option_context('display.max_rows', None): 
            if dataset == "phynet":
                df = df.loc[df["size_kb"] <= 24].sort_values(by=["accuracy"])
            print(df[["dataset", "pruning", "LR", "n_estimators", "max_leafs", "l1", "rho", "K", "train_time_sec", "train_accuracy","size_kb","n_nodes","f1", "accuracy"]])
        
        all_df.append(df)
    
    df = pd.concat(all_df)
    df.to_csv("{}.csv".format(datetime.now().strftime("%d-%m-%Y-%H:%M")),index=False)

    sdf = pd.DataFrame(statistics)
    sdf.to_csv("datasets_{}.csv".format(datetime.now().strftime("%d-%m-%Y-%H:%M")), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
    parser.add_argument("--max_leaves", help="Maximum number of leaf nodes. Can be a list of arguments for multiple experiments", nargs='+', type=int, default=[1024])
    parser.add_argument("-d", "--dataset", help="Dataset used for experiments. Can be a list of arguments for multiple dataset. Have a look at datasets.py for all supported datasets.",type=str, default=["magic"], nargs='+')
    parser.add_argument("-M", "--n_estimators", help="Number of estimators in the forest.", nargs='+', type=int, default=[32,256])
    parser.add_argument("-x", "--xval", help="Number of cross-validation runs if the dataset does not contain a train/test split.",type=int, default=5)
    parser.add_argument("-t", "--tmpdir", help="Temporary folder in which datasets should be stored.",type=str, default=None)
    parser.add_argument("-l", "--l1_reg", help="L1 regs.",nargs='+', default=[1e-3], type=float)
    parser.add_argument("--debug", help="Execute all experiments one by one with better stack traces.", action='store_true')
    args = parser.parse_args()
    
    main(args)