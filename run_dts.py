#!/usr/bin/env python3

import sys
import os
import numpy as np
from numpy.lib.arraysetops import isin
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

from sklearn.metrics import accuracy_score, hinge_loss

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
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
from datasets import get_dataset

def get_all_points(model, X):
    n_features  = X.shape[1]
    Xnew = []
    region = [ [min(X[:, i]),max(X[:, i])] for i in range(n_features)]

    stack = [ (0, region) ]  # start with the root node id (0)
    while len(stack) > 0:
        node_id, cur_region = stack.pop()
        
        is_split_node = model.tree_.children_left[node_id] != model.tree_.children_right[node_id]
        if is_split_node:
            f = model.tree_.feature[node_id]
            t = model.tree_.threshold[node_id]
            l_region = copy.deepcopy(cur_region)
            r_region = copy.deepcopy(cur_region)
                
            l_region[f][1] = min(l_region[f][1], t)
            r_region[f][0] = max(r_region[f][0], t)
                
            stack.append((model.tree_.children_left[node_id], l_region))
            stack.append((model.tree_.children_right[node_id], r_region))
        else:
            tmp = None
            for fmin, fmax in cur_region:
                features = np.linspace(start = fmin, stop = fmax, num = 10)
                # print(features)
                # print(features.shape)
                if tmp is None:
                    tmp = features
                else:
                    tmp = np.c_[ tmp, features ]
                    # XNew[:,:-1] = features
            # for _ in range(5):
            #     x = []
            #     for fmin, fmax in cur_region:
            #         x.append(np.random.uniform(low=fmin, high=fmax))
            #         #x.append( (fmin + fmax) / 2.0)
            #     Xnew.append(x)

            # x = []
            # for fmin, fmax in cur_region:
            #     x.append( fmin )
            # Xnew.append(x)

            # x = []
            # for fmin, fmax in cur_region:
            #     x.append( fmax )
            Xnew.append(tmp)
    return np.concatenate(Xnew)
    # return np.array(Xnew)

def eval_model(X, Y, idx, rfs = None, n_estimators = 64, rf_height = None, dt_height = None):
    metrics = {
        "test_accuracy":[],
        "train_accuracy":[],
        "effective_height":[],
        "n_leaves":[],
        "n_nodes":[]
    }

    models = []
    for i, (itrain, itest) in enumerate(idx):
        XTrain, YTrain = X[itrain], Y[itrain]
        XTest, YTest = X[itest], Y[itest]

        if rfs is None:
            rf = RandomForestClassifier(n_estimators = n_estimators, bootstrap = True, max_depth = rf_height, n_jobs = args.n_jobs)
        
            # scaler = MinMaxScaler()
            # XTrain = scaler.fit_transform(XTrain)
            # XTest = scaler.transform(XTest)
            rf.fit(XTrain,YTrain)
            XNewTrain = np.concatenate( [XTrain] + [get_all_points(e, X) for e in rf.estimators_] )
            print("Generated {} new training data".format(XNewTrain.shape))
            YNewTrain = rf.predict(XNewTrain) 
            model = DecisionTreeClassifier(max_depth = dt_height)
            model.fit(XNewTrain, YNewTrain)
            
            models.append(rf)
            model.estimators_ = [model]
        else:
            model = rfs[i]
            models.append(model)
        
        metrics["test_accuracy"].append(100.0*accuracy_score(model.predict_proba(XTest).argmax(axis=1), YTest))
        metrics["train_accuracy"].append(100.0*accuracy_score(model.predict_proba(XTrain).argmax(axis=1), YTrain))

        e_height = 0
        n_leaves = 0
        n_nodes = 0
        for e in model.estimators_:
            n_nodes += e.tree_.node_count
            max_depth = 0
            stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
            while len(stack) > 0:
                node_id, depth = stack.pop()

                if depth > max_depth:
                    max_depth = depth

                is_split_node =  e.tree_.children_left[node_id] != e.tree_.children_right[node_id]
                if is_split_node:
                    stack.append((e.tree_.children_left[node_id], depth + 1))
                    stack.append((e.tree_.children_right[node_id], depth + 1))
                else:
                    n_leaves += 1
            e_height += max_depth
        metrics["effective_height"].append(e_height / len(model.estimators_))
        metrics["n_leaves"].append(n_leaves / len(model.estimators_))
        metrics["n_nodes"].append(n_nodes)

    results = {
        "test_accuracy":np.mean(metrics["test_accuracy"]),
        "train_accuracy":np.mean(metrics["train_accuracy"]),
        "effective_height":np.mean(metrics["effective_height"]),
        "n_leaves":np.mean(metrics["n_leaves"]),
        "n_nodes":np.sum(metrics["n_nodes"])
    }
    return results, models
        
def main(args):

    for dataset in args.dataset:
        X, Y = get_dataset(dataset)
        
        if X is None or Y is None: 
            exit(1)

        np.random.seed(12345)

        # scaler = MinMaxScaler()
        # X = scaler.fit_transform(X)
        kf = StratifiedKFold(n_splits=args.xval, random_state=12345, shuffle=True)
        idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X, Y)], dtype=object)

        from collections import Counter
        print("Data: ", X.shape, " ", X[0:2,:])
        print("Labels: ", Y.shape, " ", Counter(Y))

        metrics = []
        
        if args.rf_height is None or len(args.rf_height) == 0:
            args.rf_height = [None]
        
        if args.dt_height is None or len(args.dt_height) == 0:
            args.dt_height = [None]

        for rh in args.rf_height:
            for dh in args.dt_height:
                print("Training initial RFs with rf = {} and dt = {}".format(rh, dh))

                results, rfs = eval_model(X, Y, idx, None, args.n_estimators, rh, dh)
                metrics.append(
                    {
                        **results,
                        "height":dh,
                        "rf_height":rh,
                        "method":"DT",
                        "K":1,
                        "dataset":dataset,
                        "base":"DT"
                    }
                )

                results, rfs = eval_model(X,Y,idx,rfs, args.n_estimators, rh, dh)
                metrics.append(
                    {
                        **results,
                        "height":rh,
                        "rf_height":rh,
                        "K":args.n_estimators,
                        "method":"RF",
                        "dataset":dataset,
                        "base":"RF"
                    }
                )
            
        df = pd.DataFrame(metrics)
        df = df.sort_values(by=["dataset","height", "method", "K"])
        grouped = df.groupby(by=["base"])
        for key, dff in grouped:
            print(dff[["dataset", "method", "height", "effective_height", "n_leaves", "K", "test_accuracy", "train_accuracy", "n_nodes"]])
    #df.to_csv("bagging_experiment.csv",index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
    parser.add_argument("-b", "--base", help="Base learner ued for experiments. Can be {{RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, HeterogenousForest}}",type=str, nargs='+', default=["RandomForestClassifier"])
    parser.add_argument("--rf_height", help="Maximum height of the trees in the base RF. Corresponds to sci-kit learns max_depth parameter. Can be a list of arguments for multiple experiments. Important: Values <= 0 are interpreted as `None` (unlimited tree depth)", nargs='+', type=int)
    parser.add_argument("--dt_height", help="Maximum height of the signle DT. Corresponds to sci-kit learns max_depth parameter. Can be a list of arguments for multiple experiments. Important: Values <= 0 are interpreted as `None` (unlimited tree depth)", nargs='+', type=int)
    parser.add_argument("-d", "--dataset", help="Dataset used for experiments",type=str, default=["wine-quality"], nargs='+')
    parser.add_argument("-n", "--n_estimators", help="Number of estimators trained for the base learner.", type=int, default=64)
    parser.add_argument("-x", "--xval", help="Number of X-val runs",type=int, default=5)
    parser.add_argument("-t", "--timeout", help="Maximum number of seconds per run. If the runtime exceeds the provided value, stop execution",type=int, default=6000)
    args = parser.parse_args()

    main(args)