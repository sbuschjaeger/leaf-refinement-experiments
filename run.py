#!/usr/bin/env python3

from datasets import get_dataset
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import random
from scipy.io.arff import loadarff
import copy

from functools import partial

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.metrics import roc_auc_score

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from experiment_runner.experiment_runner import run_experiments, Variation, generate_configs

from PyPruning.RandomPruningClassifier import RandomPruningClassifier
from PyPruning.ProxPruningClassifier import ProxPruningClassifier
from PyPruning.PruningClassifier import PruningClassifier 
from PyPruning.Papers import create_pruner 
from HeterogenousForest import HeterogenousForest

# As a baseline we also want to evaluate the unpruned classifier. To use the
# same code base below we implement a NotPruningPruner which does not prune at all
# and uses the original model. 
class NotPruningPruner(PruningClassifier):

    def __init__(self, n_estimators = 5):
        super().__init__()
        self.n_estimators = n_estimators

    def prune_(self, proba, target, data = None):
        n_received = len(proba)
        return range(0, n_received), [1.0 / n_received for _ in range(n_received)]

def pre(cfg):
    if cfg["model"] in ["RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier", "HeterogenousForest"]:
    #    model_ctor = NotPruningPruner
    #elif cfg["model"] == "RandomPruningClassifier":
        model_ctor = RandomPruningClassifier
    elif cfg["model"] == "AdaBoostClassifier":
        model_ctor = AdaBoostClassifier
    elif cfg["model"] == "ProxPruningClassifier":
        model_ctor = ProxPruningClassifier
    else:
        model_ctor = partial(create_pruner, method = cfg["model"]) 

    model_params = cfg["model_params"]

    if "out_path" in model_params and model_params["out_path"] is not None:
        model_params["out_path"] = cfg["out_path"]

    model = model_ctor(**model_params)
    return model

def fit(cfg, from_pre):
    model = from_pre
    i = cfg["run_id"]
    pruneidx = cfg["pruneidx"][i]
    X, Y = cfg["X"][pruneidx],cfg["Y"][pruneidx]
    estimators = cfg["estimators"][i]
    # print("Received {} estimators. Now pruning.".format(len(estimators)))

    if model is None:
        print(cfg)

    if cfg["model"] == "AdaBoostClassifier":
        model.fit(X, Y)
    else:
        model.prune(X, Y, estimators, cfg["classes"][i], cfg["n_classes"])
    return model

def post(cfg, from_fit):
    scores = {}
    model = from_fit
    i = cfg["run_id"]
    testidx = cfg["testidx"][i]
    X, Y = cfg["X"][testidx],cfg["Y"][testidx]

    pred = from_fit.predict_proba(X)
    scores["accuracy"] = 100.0 * accuracy_score(Y, pred.argmax(axis=1))
    # if(pred.shape[1] == 2):
    #     scores["roc_auc"] = roc_auc_score(Y, pred.argmax(axis=1))
    # else:
    #     scores["roc_auc"] = roc_auc_score(Y, pred, multi_class="ovr")
    n_total_comparisons = 0
    for est in from_fit.estimators_:
        n_total_comparisons += est.decision_path(X).sum()
    scores["avg_comparisons_per_tree"] = n_total_comparisons / (X.shape[0] * len(from_fit.estimators_))
    scores["n_nodes"] = sum( [ est.tree_.node_count for est in from_fit.estimators_] )
    scores["n_estimators"] = len(from_fit.estimators_)
    return scores


def main(args):
    if args.nl is None:
        args.nl = [None]
    else:
        args.nl = [None if h <= 0 else h for h in args.nl]

    if args.n_prune is None:
        args.n_prune = [32]

    for dataset in args.dataset:

        if args.n_jobs == 1:
            basecfg = {
                "out_path":os.path.join(dataset, "results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
                "pre": pre,
                "post": post,
                "fit": fit,
                "backend": "single",
                "verbose":True,
                "timeout":args.timeout
            }
        else:
            basecfg = {
                "out_path":os.path.join(dataset, "results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
                "pre": pre,
                "post": post,
                "fit": fit,
                "backend": "multiprocessing",
                "num_cpus":args.n_jobs,
                "verbose":True,
                "timeout":args.timeout
            }

        models = []
        print("Loading {}".format(dataset))

        X, Y = get_dataset(dataset)
        
        if X is None or Y is None: 
            exit(1)
            
        np.random.seed(12345)

        kf = StratifiedKFold(n_splits=args.xval, random_state=12345, shuffle=True)
        
        if dataset == "mnist":
            idx = np.array( [ (list(range(1000,7000)), list(range(0,1000))) ] , dtype=object)
        elif dataset == "ida2016":
            idx = np.array( [ (list(range(0,60000)), list(range(60000,76000)))  ] , dtype=object)
        elif dataset == "dota2":
            idx = np.array( [ (list(range(0,92650)), list(range(92650,102944)))  ] , dtype=object)
        else:
            idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X, Y)], dtype=object)

        from collections import Counter
        print("Data: ", X.shape, " ", X[0:2,:])
        print("Labels: ", Y.shape, " ", Counter(Y))
        break
        # HeterogenousForest requires a list of heights so pack it into another list
        for base in args.base:
            # if base == "HeterogenousForest":
            #     heights = [args.height]
            # else:
            #     heights = args.height

            for max_l in args.nl:
                print("Training initial {} with max_leaf_nodes = {}".format(base, max_l))

                trainidx = []
                testidx = []
                pruneidx = []

                # rf_estimators = []
                # rf_classes = []

                estimators = []
                classes = []
                for itrain, itest in idx:
                    if base == "RandomForestClassifier":
                        #base_model = RandomForestClassifier(n_estimators = args.n_estimators, bootstrap = True, max_depth = h, n_jobs = args.n_jobs)
                        base_model = RandomForestClassifier(n_estimators = args.n_estimators, bootstrap = True, max_leaf_nodes = max_l, n_jobs = args.n_jobs)
                    elif base == "ExtraTreesClassifier":
                        base_model = ExtraTreesClassifier(n_estimators = args.n_estimators, bootstrap = True,  max_leaf_nodes = max_l, n_jobs = args.n_jobs)
                    elif base == "BaggingClassifier":
                        # This pretty much fits a RF with max_features = None / 1.0
                        base_model = BaggingClassifier(base_estimator = DecisionTreeClassifier( max_leaf_nodes = max_l),n_estimators = args.n_estimators, bootstrap = True, n_jobs = args.n_jobs)
                    else:
                        base_model = HeterogenousForest(base=DecisionTreeClassifier, n_estimators = args.n_estimators, max_leaf_nodes = max_l, splitter = "random", bootstrap=True, n_jobs = args.n_jobs)
                    
                    if args.use_prune:
                        XTrain, _, YTrain, _, tmp_train, tmp_prune = train_test_split(X[itrain], Y[itrain], itrain, test_size = 0.33)
                        trainidx.append(tmp_train)
                        pruneidx.append(tmp_prune)
                    else:
                        XTrain, YTrain = X[itrain], Y[itrain]
                        trainidx.append(itrain)
                        pruneidx.append(itrain)
                    
                    testidx.append(itest)
                    base_model.fit(XTrain, YTrain)
                    estimators.append(copy.deepcopy(base_model.estimators_))
                    classes.append(base_model.classes_)

                experiment_cfg = {
                    "dataset":dataset,
                    "X":X,
                    "Y":Y,
                    "trainidx":trainidx,
                    "testidx":testidx,
                    "pruneidx":pruneidx,
                    "repetitions":1 if dataset in ["mnist", "ida2016", "dota2"] else args.xval,
                    "seed":12345,
                    "estimators":estimators,
                    "classes":classes,
                    "n_classes":len(set(Y)),
                    "max_leaf_nodes" : max_l,
                    "base":base
                }

                for K in args.n_prune:
                    models.append(
                        {
                            "model":base,
                            "model_params":{
                                "n_estimators":K
                            },
                            **experiment_cfg
                        }
                    )

                    # models.append(
                    #     {
                    #         "model":"AdaBoostClassifier",
                    #         "model_params":{
                    #             "n_estimators":K,
                    #             "base_estimator":DecisionTreeClassifier(max_leaf_nodes = max_l)
                    #         },
                    #         **experiment_cfg
                    #     }
                    # )

                    for model in ["individual_contribution", "individual_error",  "reduced_error", "complementariness"]:
                        models.append(
                            {
                                "model":model,
                                "model_params":{
                                    "n_estimators":K,
                                    "n_jobs":1
                                },
                                **experiment_cfg
                            }
                        )

                    models.append(
                        {
                            "model":"cluster_accuracy",
                            "model_params":{
                                "n_estimators":K,
                                "cluster_options":{
                                    "n_jobs":1
                                }
                            },
                            **experiment_cfg
                        }
                    )

                    models.append(
                        {
                            "model":"largest_mean_distance",
                            "model_params":{
                                "n_estimators":K,
                                "selector_options":{
                                    "n_jobs":1
                                }
                            },
                            **experiment_cfg
                        }
                    )

                    rho = [0.25,0.3,0.35,0.4,0.45,0.5]
                    for r in rho:
                        models.append(
                            {
                                "model":"drep",
                                "model_params":{
                                    "n_estimators":K,
                                    "metric_options": {
                                        "rho": r
                                    },
                                    "n_jobs":1
                                },
                                **experiment_cfg
                            }
                        ) 

                    tmp_cfg = copy.deepcopy(experiment_cfg)
                    for i in range(len(tmp_cfg["estimators"])):
                        tmp_cfg["estimators"][i] = tmp_cfg["estimators"][i][0:K]

                    models.append(
                        {
                            "model":"ProxPruningClassifier",
                            "model_params":{
                                "ensemble_regularizer":"L0",
                                "l_ensemble_reg":0,
                                "l_tree_reg":0,
                                "batch_size" : 128,
                                "epochs": 50,
                                "step_size": 1e-1, 
                                "verbose":False,
                                "loss":"mse",
                                "update_leaves":True,
                                "normalize_weights":True,
                                "update_weights":False
                            },
                            **tmp_cfg
                        }
                    )

        random.shuffle(models)
        run_experiments(basecfg, models)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
    parser.add_argument("-b", "--base", help="Base learner ued for experiments. Can be {{RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, HeterogenousForest}}",type=str, nargs='+', default=["RandomForestClassifier"])
    parser.add_argument("--nl", help="Maximum number of leaf nodes (corresponds to scikit-learns max_leaf_nodes parameter)", nargs='+', type=int)
    parser.add_argument("-d", "--dataset", help="Dataset used for experiments",type=str, default=["wine-quality"], nargs='+')
    parser.add_argument("-n", "--n_estimators", help="Number of estimators trained for the base learner.", type=int, default=64)
    parser.add_argument("-K", "--n_prune", help="Size of the pruned ensemble. Can be a list for multiple experiments.",nargs='+', type=int, default=[32])
    parser.add_argument("-x", "--xval", help="Number of X-val runs",type=int, default=5)
    parser.add_argument("-p", "--use_prune", help="Use a train / prune / test split. If false, the training data is also used for pruning", action="store_true", default=False)
    parser.add_argument("-t", "--timeout", help="Maximum number of seconds per run. If the runtime exceeds the provided value, stop execution",type=int, default=6000)
    args = parser.parse_args()

    main(args)