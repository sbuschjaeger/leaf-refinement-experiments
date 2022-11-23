#!/usr/bin/env python3

import os
from enum import unique
from operator import index
import time
import pandas as pd
import copy
import numpy as np
import argparse
from multiprocessing import Pool
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from random import shuffle
from PyPruning.Papers import create_pruner 

from sklearn.model_selection import StratifiedKFold

from sklearn.utils import parallel_backend

from LeafRefinement import LeafRefinery

from Datasets import get_dataset

# from Metrics import accuracy, avg_accuracy, avg_rademacher, c_bound, n_nodes, n_leaves, effective_height, soft_hinge, mse, bias, diversity

from sklearn.metrics import accuracy_score, f1_score
import fastinference.Loader# import Loader  #.models.Ensemble import Ensemble

def get_n_estimators(model):
    return len(model.trees) if hasattr(model, "trees") else len(model.estimators_)

def get_n_nodes(model):
    return sum([e.n_nodes for e in model.trees]) if hasattr(model, "trees") else sum([e.tree_.node_count for e in model.estimators_])

def get_size_kb(model):
    n_nodes = get_n_nodes(model)
    return n_nodes * (17 + 4*model.n_classes_ )  / 1024.0

def implement(cfg, X_train, Y_train, X_test, Y_test, out_path):
    rf = cfg["base"]
    rf.fit(X_train, Y_train)

    model = cfg["model"]
    model.prune(X_train, Y_train, rf.estimators_, rf.classes_, rf.n_classes_)

    dummy = AdaBoostClassifier(n_estimators=len(model.trees))
    dummy.estimators_ = [e.model for e in model.trees]
    dummy.estimator_weights_ = model.weights
    dummy.classes_ = rf.classes_
    dummy.n_classes_ = rf.n_classes_
    dummy.n_features_in_ = rf.n_features_in_

    acc = accuracy_score(model.predict(X_test),Y_test)*100.0
    fimodel = fastinference.Loader.model_from_sklearn(dummy, name = cfg["name"], accuracy = acc)

    if not os.path.exists(os.path.join(out_path, cfg["name"])):
        os.makedirs(os.path.join(out_path, cfg["name"]))

    fimodel.optimize("quantize", {"quantize_splits" :  "rounding", "quantize_leafs" : 1000}, None, {})
    fimodel.implement(os.path.join(out_path, cfg["name"]), cfg["name"], "cpp", "cpp.native", feature_type="int", label_type="int")

    return acc, get_n_estimators(model), get_n_nodes(model), get_size_kb(model), f1_score(Y_test, model.predict(X_test), average="macro")

# ./run.py -j 42 -x 5 -l 1.0465 1.0466 1.0467 1.0467 1.0468  -M 2 4 8 16 256 --max_leafs 12 -d phynet
def main(args):
    X_train,Y_train,X_test,Y_test = get_dataset("phynet", args.tmpdir)

    configs = [
        # {
        #     "name":"RE",
        #     "model": LeafRefinery(
        #         epochs = 0, 
        #         lr = 0, 
        #         batch_size = 1024, 
        #         optimizer = "adam", 
        #         verbose = args.debug,
        #         loss_function = "mse", 
        #         loss_type = "upper", 
        #         l_reg = 1.0,
        #         l1_strength = 0,
        #         leaf_refinement=False,
        #         pruner = create_pruner(method = "reduced_error", n_estimators = 8)
        #     ),
        #     "base":RandomForestClassifier(n_estimators=256,  max_leaf_nodes=8, bootstrap = True, random_state=42) 
        # },
        # {
        #     "name":"LMD",
        #     "model": LeafRefinery(
        #         epochs = 0, 
        #         lr = 0, 
        #         batch_size = 1024, 
        #         optimizer = "adam", 
        #         verbose = args.debug,
        #         loss_function = "mse", 
        #         loss_type = "upper", 
        #         l_reg = 1.0,
        #         l1_strength = 0,
        #         leaf_refinement=False,
        #         pruner = create_pruner(method = "largest_mean_distance", n_estimators = 4)
        #     ),
        #     "base":RandomForestClassifier(n_estimators=256,  max_leaf_nodes=12, bootstrap = True, random_state=42) 
        # },
        # {
        #     "name":"RF",
        #     "model": LeafRefinery(
        #         epochs = 0, 
        #         lr = 0, 
        #         batch_size = 1024, 
        #         optimizer = "adam", 
        #         verbose = args.debug,
        #         loss_function = "mse", 
        #         loss_type = "upper", 
        #         l_reg = 1.0,
        #         l1_strength = 0,
        #         leaf_refinement=False,
        #         pruner = create_pruner(method = "random", n_estimators = 4)
        #     ),
        #     "base":RandomForestClassifier(n_estimators=256,  max_leaf_nodes=12, bootstrap = True, random_state=42) 
        # },
        # {
        #     "name":"LR",
        #     "model": LeafRefinery(
        #         epochs = 20, 
        #         lr = 1e-2, 
        #         batch_size = 1024, 
        #         optimizer = "adam", 
        #         verbose = args.debug,
        #         loss_function = "mse", 
        #         loss_type = "upper", 
        #         l_reg = 1.0,
        #         l1_strength = 0,
        #         leaf_refinement=True,
        #         pruner = create_pruner(method = "random", n_estimators = 4)
        #     ),
        #     "base":RandomForestClassifier(n_estimators=256,  max_leaf_nodes=12, bootstrap = True, random_state=42) 
        # },
        # {
        #     "name":"IE",
        #     "model": LeafRefinery(
        #         epochs = 0, 
        #         lr = 0, 
        #         batch_size = 1024, 
        #         optimizer = "adam", 
        #         verbose = args.debug,
        #         loss_function = "mse", 
        #         loss_type = "upper", 
        #         l_reg = 1.0,
        #         l1_strength = 0,
        #         leaf_refinement=False,
        #         pruner = create_pruner(method = "individual_error", n_estimators = 4)
        #     ),
        #     "base":RandomForestClassifier(n_estimators=256,  max_leaf_nodes=12, bootstrap = True, random_state=42) 
        # },
        # {
        #     "name":"IC",
        #     "model": LeafRefinery(
        #         epochs = 0, 
        #         lr = 0, 
        #         batch_size = 1024, 
        #         optimizer = "adam", 
        #         verbose = args.debug,
        #         loss_function = "mse", 
        #         loss_type = "upper", 
        #         l_reg = 1.0,
        #         l1_strength = 0,
        #         leaf_refinement=False,
        #         pruner = create_pruner(method = "individual_contribution", n_estimators = 8)
        #     ),
        #     "base":RandomForestClassifier(n_estimators=256,  max_leaf_nodes=8, bootstrap = True, random_state=42) 
        # },
        # {
        #     "name":"COMP",
        #     "model": LeafRefinery(
        #         epochs = 0, 
        #         lr = 0, 
        #         batch_size = 1024, 
        #         optimizer = "adam", 
        #         verbose = args.debug,
        #         loss_function = "mse", 
        #         loss_type = "upper", 
        #         l_reg = 1.0,
        #         l1_strength = 0,
        #         leaf_refinement=False,
        #         pruner = create_pruner(method = "complementariness", n_estimators = 4)
        #     ),
        #     "base":RandomForestClassifier(n_estimators=256,  max_leaf_nodes=12, bootstrap = True, random_state=42) 
        # },
        # {
        #     "name":"DREP",
        #     "model": LeafRefinery(
        #         epochs = 20, 
        #         lr = 1e-2, 
        #         batch_size = 1024, 
        #         optimizer = "adam", 
        #         verbose = args.debug,
        #         loss_function = "mse", 
        #         loss_type = "upper", 
        #         l_reg = 1.0,
        #         l1_strength = 0,
        #         leaf_refinement=False,
        #         pruner = create_pruner(method = "drep", n_estimators = 8, **{"metric_options":{"rho":0.45}})
        #     ),
        #     "base":RandomForestClassifier(n_estimators=256,  max_leaf_nodes=8, bootstrap = True, random_state=42) 
        # },
        {
            "name":"L1+LR",
            "model": LeafRefinery(
                epochs = 20, 
                lr = 1e-2, 
                batch_size = 1024, 
                optimizer = "adam", 
                verbose = args.debug,
                loss_function = "mse", 
                loss_type = "upper", 
                l_reg = 1.0,
                #l1_strength = 1.0468, #1.25,
                l1_strength = 1.047,
                leaf_refinement=True,
                pruner = "L1"
            ),
            # 6
            "base":RandomForestClassifier(n_estimators=256,  max_leaf_nodes=12, bootstrap = True, random_state=42) 
        },
        {
            "name":"L1",
            "model": LeafRefinery(
                epochs = 20, 
                lr = 1e-2, 
                batch_size = 1024, 
                optimizer = "adam", 
                verbose = args.debug,
                loss_function = "mse", 
                loss_type = "upper", 
                l_reg = 1.06,
                l1_strength = 1.00,
                leaf_refinement=False,
                pruner = "L1"
            ),
            # 6
            "base":RandomForestClassifier(n_estimators=256,  max_leaf_nodes=12, bootstrap = True, random_state=42) 
        }
    ]

    accs = []
    for cfg in configs:
        print("Implementing {}".format(cfg["name"]))
        acc,n_est,n_node,size,f1 = implement(cfg, X_train,Y_train,X_test,Y_test, args.out)
        accs.append({
            "name":cfg["name"],
            "accuracy":acc,
            "No. nodes":n_node,
            "No. trees":n_est,
            "Estimated size [KB]":size,
            "f1":f1
        })

    df = pd.DataFrame(accs)
    df = df.sort_values(by=["name"])
    df = df[["name", "accuracy", "f1", "Estimated size [KB]"]].T
    print(df)
    tmp = df.to_latex(na_rep="-")#float_format="%.2f", index=False
    print(tmp)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", help="Outpath in which files should be written",type=str, default=os.path.join("phynet", "models"))
    parser.add_argument("-t", "--tmpdir", help="Temporary folder in which datasets should be stored.",type=str, default=None)
    parser.add_argument("--debug", help="Get more output for debugging the models.", action='store_true')


    args = parser.parse_args()
    
    main(args)