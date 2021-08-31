#!/usr/bin/env python3

from PyPruning.NCPruningClassifier import NCPruningClassifier
from OversampleForest import OversampleForest
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
from PyPruning.ProxPruningClassifier import ProxPruningClassifier, avg_path_len_regularizer
from PyPruning.PruningClassifier import PruningClassifier 
from PyPruning.Papers import create_pruner 
from HeterogenousForest import HeterogenousForest
from datasets import get_dataset

def loss(model, X, target):
    base_preds = []
    if hasattr(model, "_individual_proba"):
        base_preds = model._individual_proba(X)
        base_preds = np.array([w * p for p,w in zip(base_preds, model.weights_)])
        # fbar = np.sum(scaled_prob,axis=0)
    else:
        for e in model.estimators_:
            base_preds.append( e.predict_proba(X) )
    fbar = np.mean(base_preds,axis=0)
    target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(model.n_classes_)] for y in target] )
    
    # from scipy.special import softmax
    # p = softmax(fbar, axis=1)
    # b = -target_one_hot*np.log(p + 1e-7)
    # return np.sum(np.mean(b,axis=1))
    return ((fbar - target_one_hot)**2).mean()

def bias(model, X, target):
    target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(model.n_classes_)] for y in target] )
    base_preds = []
    if hasattr(model, "_individual_proba"):
        base_preds = model._individual_proba(X)
        base_preds = np.array([w * p for p,w in zip(base_preds, model.weights_)])
        # fbar = np.sum(scaled_prob,axis=0)
    else:
        for e in model.estimators_:
            base_preds.append( e.predict_proba(X) )

    biases = []
    for hpred in base_preds:
        b = ((hpred - target_one_hot)**2).mean()
        
        # from scipy.special import softmax
        # p = softmax(hpred, axis=1)
        # b = -target_one_hot*np.log(p + 1e-7)
        # b = np.sum(np.mean(b,axis=1))
        biases.append(b)
    return np.mean(biases)

def min_max_accuracy(model, X, target, prefix="test"):
    base_preds = []
    if hasattr(model, "_individual_proba"):
        base_preds = model._individual_proba(X)
        base_preds = np.array([len(model.estimators_) * w * p for p,w in zip(base_preds, model.weights_)])
    else:
        for e in model.estimators_:
            base_preds.append( e.predict_proba(X) )

    accs = []
    for hpred in base_preds:
        accs.append(100.0*accuracy_score(hpred.argmax(axis=1), target))

    return {
        "min_" + prefix + "_accuracy" : min(accs),
        "max_" + prefix + "_accuracy" : max(accs),
        "mean_" + prefix + "_accuracy" : np.mean(accs),
        "std_" + prefix + "_accuracy" : np.std(accs)
    }

def eval_model(X, Y, idx, model, rfs = None, use_prune = False, eval_bases = False):
    metrics = {
        "test_accuracy":[],
        "test_bias":[],
        "test_diversity":[],
        "test_loss":[],
        "train_accuracy":[],
        "train_bias":[],
        "train_diversity":[],
        "train_loss":[],
        "effective_height":[],
        "n_leaves":[],
        "n_nodes":[],
        "n_trees":[]
    }

    models = []
    for i, (itrain, itest) in enumerate(idx):
        imodel = copy.deepcopy(model)

        if use_prune and not eval_bases:
            XTrain, YTrain = X[itrain], Y[itrain]

            tmp = [XTrain]
            for _ in range(5):
                tmp.append( XTrain + np.random.normal(loc = 0.0, scale = 0.1, size = XTrain.shape) )
            XPrune = np.concatenate(tmp)
            YPrune = rfs[i].predict_proba(XPrune).argmax(axis=1)
            print("Generated {} pruning data".format(XPrune.shape))

            XTest, YTest = X[itest], Y[itest]
            # XTrain, XPrune, YTrain, YPrune = train_test_split(X[itrain], Y[itrain], test_size = 0.25)
            # XTest, YTest = X[itest], Y[itest]
        else:
            XTrain, YTrain = X[itrain], Y[itrain]
            XPrune, YPrune = XTrain, YTrain
            XTest, YTest = X[itest], Y[itest]
        
        if not eval_bases:
            imodel.prune(XPrune, YPrune, rfs[i].estimators_, rfs[i].classes_, rfs[i].n_classes_)
        else:
            imodel.fit(XTrain, YTrain)
        
        models.append(imodel)

        metrics["test_accuracy"].append(100.0*accuracy_score(imodel.predict_proba(XTest).argmax(axis=1), YTest))
        metrics["test_bias"].append(bias(imodel, XTest, YTest))
        metrics["test_loss"].append(loss(imodel, XTest, YTest))
        metrics["test_diversity"].append(metrics["test_bias"][-1] - metrics["test_loss"][-1])
        metrics = {
            **min_max_accuracy(imodel, XTest, YTest, "test"),
            **metrics
        }
        metrics["train_accuracy"].append(100.0*accuracy_score(imodel.predict_proba(XTrain).argmax(axis=1), YTrain))

        e_height = 0
        n_leaves = 0
        n_nodes = 0
        for e in imodel.estimators_:
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
            # TODO Mit n_samples Gewichten
            e_height += max_depth
        metrics["effective_height"].append(e_height / len(imodel.estimators_))
        metrics["n_leaves"].append(n_leaves / len(imodel.estimators_))
        metrics["n_nodes"].append(n_nodes)
        metrics["n_trees"].append(len(imodel.estimators_))

    results = {
        "test_accuracy":np.mean(metrics["test_accuracy"]),
        "test_bias":np.mean(metrics["test_bias"]),
        "test_diversity":np.mean(metrics["test_diversity"]),
        "test_loss":np.mean(metrics["test_loss"]),
        "max_test_accuracy":np.mean(metrics["max_test_accuracy"]),
        "min_test_accuracy":np.mean(metrics["min_test_accuracy"]),
        "mean_test_accuracy":np.mean(metrics["mean_test_accuracy"]),
        "std_test_accuracy":np.mean(metrics["std_test_accuracy"]),
        "train_accuracy":np.mean(metrics["train_accuracy"]),
        "effective_height":np.mean(metrics["effective_height"]),
        "n_leaves":np.mean(metrics["n_leaves"]),
        "n_nodes":np.sum(metrics["n_nodes"]),
        "n_trees":np.sum(metrics["n_trees"])
    }
    return results, models
        
class FirstKPruner(PruningClassifier):
    def __init__(self, n_estimators = 5):
        super().__init__()
        self.n_estimators = n_estimators

    def prune_(self, proba, target, data = None):
        return range(0, self.n_estimators), [1.0 / self.n_estimators for _ in range(self.n_estimators)]

def main(args):

    for dataset in args.dataset:
        X, Y = get_dataset(dataset)
        
        if X is None or Y is None: 
            exit(1)

        np.random.seed(12345)

        kf = StratifiedKFold(n_splits=args.xval, random_state=12345, shuffle=True)
        idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X, Y)], dtype=object)

        from collections import Counter
        print("Data: ", X.shape, " ", X[0:2,:])
        print("Labels: ", Y.shape, " ", Counter(Y))

        metrics = []

        for h in args.height:
            print("Training initial RFs with h = {}".format(h))
            #rf = RandomForestClassifier(n_estimators = args.n_estimators, bootstrap = True, max_depth = h, n_jobs = args.n_jobs)
            rf = ExtraTreesClassifier(n_estimators = args.n_estimators, bootstrap = True, max_depth = h, n_jobs = args.n_jobs, max_features=1, max_samples=0.25)
            results, rfs = eval_model(X,Y,idx,rf,None,args.use_prune,True)
            metrics.append(
                {
                    **results,
                    "height":h,
                    "K":args.n_estimators,
                    "method":"RF-default",
                    "dataset":dataset,
                    "base":"RF"
                }
            )

            for K in args.n_prune:
                print("Prung via PP with K = {}".format(K))
                pruning_options = {
                    "ensemble_regularizer":"hard-L0",
                    "l_ensemble_reg":K,
                    "batch_size" : 128,
                    "epochs": 15,
                    "step_size": 1e-2, 
                    "verbose":True,
                    "loss":"mse",
                    "normalize_weights":True,
                    "tree_regularizer":avg_path_len_regularizer,
                    "l_tree_reg":0.05
                }

                pp_pruner = ProxPruningClassifier(**pruning_options)
                results, _ = eval_model(X,Y,idx,pp_pruner,rfs,args.use_prune,False)
                metrics.append(
                    {
                        **results,
                        "height":h,
                        "K":K,
                        "method":"PP+RF",
                        "dataset":dataset,
                        "base":"RF"
                    }
                )

                print("Prung via RE+RF with K = {}".format(K))
                re_pruner = create_pruner("reduced_error", n_estimators = K)
                results, _ = eval_model(X,Y,idx,re_pruner,rfs,args.use_prune,False)
                metrics.append(
                    {
                        **results,
                        "height":h,
                        "K":K,
                        "method":"RE+RF",
                        "dataset":dataset,
                        "base":"RF"
                    }
                )

                print("Prung via FK+RF with K = {}".format(K))
                results, _ = eval_model(X,Y,idx,FirstKPruner(n_estimators=K),rfs,args.use_prune,False)
                metrics.append(
                    {
                        **results,
                        "height":h,
                        "K":K,
                        "method":"FK+RF",
                        "dataset":dataset,
                        "base":"RF"
                    }
                )
            #     print("Prung via RE+RF with K = {}".format(K))
            #     re_pruner = create_pruner("reduced_error", n_estimators = K)
            #     results, _ = eval_model(X,Y,idx,re_pruner,ofs,args.use_prune,False)
            #     metrics.append(
            #         {
            #             **results,
            #             "height":h,
            #             "K":K,
            #             "method":"RE+OF",
            #             "dataset":dataset,
            #             "base":"OF"
            #         }
            #     )

            #     results, _ = eval_model(X,Y,idx,FirstKPruner(n_estimators=K),ofs,args.use_prune,False)
            #     metrics.append(
            #         {
            #             **results,
            #             "height":h,
            #             "K":K,
            #             "method":"FK+OF",
            #             "dataset":dataset,
            #             "base":"OF"
            #         }
            #     )

                # print("Prung via PP with K = {}".format(K))
                # pruning_options = {
                #     "ensemble_regularizer":"hard-L0",
                #     "l_ensemble_reg":K,
                #     "l_tree_reg":0,
                #     "batch_size" : 256,
                #     "epochs": 5,
                #     "step_size": 1e-2, 
                #     "verbose":False,
                #     "loss":"mse",
                #     "update_leaves":False
                # }

                # pp_pruner = ProxPruningClassifier(**pruning_options)
                # results, _ = eval_model(X,Y,idx,pp_pruner,rfs,args.use_prune,False)
                # metrics.append(
                #     {
                #         **results,
                #         "height":h,
                #         "K":K,
                #         "method":"PP+RF",
                #         "dataset":dataset,
                #         "base":"RF"
                #     }
                # )
            
        df = pd.DataFrame(metrics)
        df = df.sort_values(by=["dataset","height", "method", "K"])
        print(df[["dataset", "method", "height", "effective_height", "n_leaves", "K", "test_accuracy", "test_bias", "test_diversity", "test_loss", "train_accuracy", "max_test_accuracy", "min_test_accuracy", "mean_test_accuracy", "std_test_accuracy", "n_nodes"]])
        # grouped = df.groupby(by=["base"])
        # for key, dff in grouped:
        #     print(dff[["dataset", "method", "height", "effective_height", "n_leaves", "K", "test_accuracy", "test_bias", "test_diversity", "test_loss", "train_accuracy", "max_test_accuracy", "min_test_accuracy", "mean_test_accuracy", "std_test_accuracy", "n_nodes"]])
    #df.to_csv("bagging_experiment.csv",index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
    parser.add_argument("-b", "--base", help="Base learner ued for experiments. Can be {{RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, HeterogenousForest}}",type=str, nargs='+', default=["RandomForestClassifier"])
    parser.add_argument("--height", help="Maximum height of the trees. Corresponds to sci-kit learns max_depth parameter. Can be a list of arguments for multiple experiments. Important: Values <= 0 are interpreted as `None` (unlimited tree depth)", nargs='+', type=int, default=[5])
    parser.add_argument("-d", "--dataset", help="Dataset used for experiments",type=str, default=["magic"], nargs='+')
    parser.add_argument("-n", "--n_estimators", help="Number of estimators trained for the base learner.", type=int, default=64)
    parser.add_argument("-T", "--n_prune", help="Size of the pruned ensemble. Can be a list for multiple experiments.",nargs='+', type=int, default=[32])
    parser.add_argument("-x", "--xval", help="Number of X-val runs",type=int, default=5)
    parser.add_argument("-p", "--use_prune", help="Use a train / prune / test split. If false, the training data is also used for pruning", action="store_true", default=False)
    parser.add_argument("-t", "--timeout", help="Maximum number of seconds per run. If the runtime exceeds the provided value, stop execution",type=int, default=6000)
    args = parser.parse_args()


    main(args)