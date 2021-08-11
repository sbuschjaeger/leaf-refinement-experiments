#!/usr/bin/env python3

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


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class ProxForestClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_params, prox_params, frac_pruning = 0.5):
        self.base_params = base_params
        self.prox_params = prox_params
        self.frac_pruning = frac_pruning

        #self.model = ExtraTreesClassifier(**base_params)
        self.model = HeterogenousForest(base=DecisionTreeClassifier,**base_params, bootstrap = True)
        self.pruner = ProxPruningClassifier(**prox_params)

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        # Xtrain, Xprune, Ytrain, Yprune = train_test_split(X, y, test_size = self.frac_pruning)
        # self.model.fit(Xtrain,Ytrain)
        # self.pruner.prune(Xprune, Yprune, self.model.estimators_, self.model.classes_, self.model.n_classes_)
        
        self.model.fit(X,y)
        self.pruner.prune(X, y, self.model.estimators_, self.model.classes_, self.model.n_classes_)

        # This is just so we can get valid statistics later
        self.estimators_ = self.pruner.estimators_
        self.weights_ = self.pruner.weights_ 
        
        # Return the classifier
        return self

    def predict_proba(self, X):
        return self.pruner.predict_proba(X)
    
    def predict(self, X):
        return self.pruner.predict(X)

def pre(cfg):
    if cfg["model"] == "RandomForestClassifier":
        return RandomForestClassifier(**cfg.pop("model_params", {}))
    elif cfg["model"] == "ProxForestClassifier":
        return ProxForestClassifier(cfg.pop("base_params", None), cfg.pop("prox_params", None), cfg.pop("frac_pruning", 0))
    else:
        return None

def fit(cfg, from_pre):
    model = from_pre
    i = cfg["run_id"]
    trainidx = cfg["trainidx"][i]
    X, Y = cfg["X"][trainidx],cfg["Y"][trainidx]
    model.fit(X,Y)
    return model

def post(cfg, from_fit):
    scores = {}
    model = from_fit
    i = cfg["run_id"]
    testidx = cfg["testidx"][i]
    trainidx = cfg["trainidx"][i]
    Xtest, Ytest = cfg["X"][testidx],cfg["Y"][testidx]
    Xtrain, Ytrain = cfg["X"][trainidx],cfg["Y"][trainidx]

    pred_test = from_fit.predict_proba(Xtest)
    pred_train = from_fit.predict_proba(Xtrain)
    scores["test_accuracy"] = 100.0 * accuracy_score(Ytest, pred_test.argmax(axis=1))
    scores["train_accuracy"] = 100.0 * accuracy_score(Ytrain, pred_train.argmax(axis=1))
    # if(pred.shape[1] == 2):
    #     scores["roc_auc"] = roc_auc_score(Y, pred.argmax(axis=1))
    # else:
    #     scores["roc_auc"] = roc_auc_score(Y, pred, multi_class="ovr")
    n_total_comparisons = 0
    for est in from_fit.estimators_:
        n_total_comparisons += est.decision_path(Xtest).sum()
    scores["avg_comparisons_per_tree"] = n_total_comparisons / (Xtest.shape[0] * len(from_fit.estimators_))
    scores["n_nodes"] = sum( [ est.tree_.node_count for est in from_fit.estimators_] )
    scores["n_estimators"] = len(from_fit.estimators_)
    return scores


def main(args):
    if args.height is None:
        args.height = [None]
    else:
        args.height = [None if h <= 0 else h for h in args.height]

    if args.n_prune is None:
        args.n_prune = [32]

    if len(args.dataset) == 1:
        outpath = args.dataset[0]
        args.dataset = args.dataset
    else:
        outpath = "multi"

    if args.n_jobs == 1:
        basecfg = {
            "out_path":os.path.join(outpath, "results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
            "pre": pre,
            "post": post,
            "fit": fit,
            "backend": "single",
            "verbose":True,
            "timeout":args.timeout
        }
    else:
        basecfg = {
            "out_path":os.path.join(outpath, "results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
            "pre": pre,
            "post": post,
            "fit": fit,
            "backend": "multiprocessing",
            "num_cpus":args.n_jobs,
            "verbose":True,
            "timeout":args.timeout
        }

    models = []
    for dataset in args.dataset:
        print("Loading {}".format(dataset))

        if dataset == "magic":
            #df = pd.read_csv(os.path.join(args.dataset, "magic04.data"))
            df = pd.read_csv(os.path.join(dataset, "magic04.data"))
            X = df.values[:,:-1].astype(np.float64)
            Y = df.values[:,-1]
            Y = np.array([0 if y == 'g' else 1 for y in Y])
        elif dataset == "covtype":
            df = pd.read_csv(os.path.join(dataset, "covtype.data"), header=None)
            X = df.values[:,:-1].astype(np.float64)
            Y = df.values[:,-1]
            Y = Y - min(Y)
        elif dataset == "letter":
            df = pd.read_csv(os.path.join(dataset, "letter-recognition.data"), header=None)
            X = df.values[:,1:].astype(np.float64)
            Y = df.values[:,0]
            Y = np.array( [ord(y) - 65 for y in Y] )
        elif dataset == "adult":
            col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                "hours-per-week", "native-country", "label"
            ]
            df = pd.read_csv(os.path.join(dataset, "adult.data"), header=None, names=col_names)
            df = df.dropna()
            label = df.pop("label")
            Y = np.array([0 if l == " <=50K" else 1 for l in label])
            df = pd.get_dummies(df)
            X = df.values
        elif dataset == "bank":
            df = pd.read_csv(os.path.join(dataset, "bank-full.csv"), header=0, delimiter=";")
            df = df.dropna()
            label = df.pop("y")
            Y = np.array([0 if l == "no" else 1 for l in label])
            df = pd.get_dummies(df)
            X = df.values
        elif dataset == "shuttle":
            df = pd.read_csv(os.path.join(dataset, "data.csv"), delimiter=" ")
            Y = df.values[:,-1]
            Y = Y - min(Y)
            Y = np.array( [1 if y > 0 else 0 for y in Y] )
            X = df.values[:,:-1]
        elif dataset == "dry-beans":
            df = pd.read_excel(os.path.join(dataset,"DryBeanDataset","Dry_Bean_Dataset.xlsx"), header = 0)
            df = df.dropna()
            label = df.pop("Class")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "spambase":
            df = pd.read_csv(os.path.join(dataset,"spambase.csv"), header = 0, delimiter=",")
            df = df.dropna()
            label = df.pop("class")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "satimage":
            df = pd.read_csv(os.path.join(dataset,"satimage.csv"), header = 0, delimiter=",")
            df = df.dropna()
            label = df.pop("class")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "connect":
            df = pd.read_csv(os.path.join(dataset,"connect.csv"), header = 0, delimiter=",")
            df = df.dropna()
            label = df.pop("class")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "mozilla":
            df = pd.read_csv(os.path.join(dataset,"mozilla.csv"), header = 0, delimiter=",")
            df = df.dropna()
            label = df.pop("state")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset in ["eeg", "elec", "nomao", "polish-bankruptcy"]:
            if dataset == "eeg":
                data, meta = loadarff(os.path.join("eeg", "EEG Eye State.arff"))
            elif dataset == "elec":
                data, meta = loadarff(os.path.join("elec", "elecNormNew.arff"))
            elif dataset == "nomao":
                data, meta = loadarff(os.path.join("nomao", "nomao.arff.txt"))
            else:
                # For nor special reason we focus on bankrupcty prediction after 1 year. Other values would also be okay
                data, meta = loadarff(os.path.join("polish-bankruptcy", "1year.arff"))

            Xdict = {}
            for cname, ctype in zip(meta.names(), meta.types()):
                # Get the label attribute for the specific dataset:
                #   eeg: eyeDetection
                #   elec: class
                #   nomao: Class
                #   polish-bankruptcy: class
                if cname in ["eyeDetection", "class",  "Class"]:
                    enc = LabelEncoder()
                    Xdict["label"] = enc.fit_transform(data[cname])
                else:
                    Xdict[cname] = data[cname]
            df = pd.DataFrame(Xdict)
            df = pd.get_dummies(df)
            df.dropna(axis=1, inplace=True)
            Y = df["label"].values.astype(np.int32)
            df = df.drop("label", axis=1)

            X = df.values.astype(np.float64)
        elif dataset == "wine-quality":
            df = pd.read_csv(os.path.join(dataset,"data.csv"), header = 0, delimiter=";")
            df = df.dropna()
            label = df.pop("quality")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "thyroid":
            df = pd.read_csv(os.path.join(dataset,"ann-train.data"), header = None, delimiter=" ")
            # For some reason there are two whitespaces at the end of each line
            label = df.values[:,-3]
            X = df.values[:,:-3]
            le = LabelEncoder()
            Y = le.fit_transform(label)
        elif dataset == "pen-digits":
            df = pd.read_csv(os.path.join(dataset,"data.txt"), header = None, delimiter=",")
            label = df.values[:,-1]
            X = df.values[:,:-1]
            le = LabelEncoder()
            Y = le.fit_transform(label)
        elif dataset == "japanese-vowels":
            df = pd.read_csv(os.path.join(dataset,"japanese-vowels.csv"), header = 0, delimiter=",")
            df = df.dropna()
            label = df.pop("speaker")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "gas-drift":
            df = pd.read_csv(os.path.join(dataset,"gas-drift.csv"), header = 0, delimiter=",")
            df = df.dropna()
            label = df.pop("Class")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "occupancy":
            df = pd.read_csv(os.path.join(dataset,"data.csv"), header = 0, delimiter=",")
            df = df.dropna()
            label = df.pop("Occupancy")
            df.pop("date")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "mnist":
            df = pd.read_csv(os.path.join(dataset,"data.csv"), header = 0, delimiter=",")
            df = df.dropna()
            label = df.values[:,0]
            X = df.values[:,1:]
            le = LabelEncoder()
            Y = le.fit_transform(label)
        elif dataset == "avila":
            df = pd.read_csv(os.path.join(dataset, "data.csv"), header=None)
            df = df.dropna()
            X = df.values[:,:-1].astype(np.float64)
            label = df.values[:,-1]
            le = LabelEncoder()
            Y = le.fit_transform(label)
        elif dataset == "weight-lifting":
            df = pd.read_csv(os.path.join(dataset, "data.csv"), skiprows=2)
            df = df.dropna(axis=1)

            # There is not documentation on these values on UCI, only that statistics are computed in a 1 second window. I assume that these attributes are required to compute the statistics, but should not be part of the ML problem. I am just ignoring those. Lets see.
            df.pop("user_name")
            df.pop("raw_timestamp_part_1")
            df.pop("raw_timestamp_part_2")
            df.pop("cvtd_timestamp")
            df.pop("new_window")
            df.pop("num_window")
            label = df.pop("classe")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "ida2016":
            df = pd.read_csv(os.path.join(dataset, "data.csv"), skiprows=20,na_values="na")
            df = df.fillna(-1)
            label = df.pop("class")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "postures":
            df = pd.read_csv(os.path.join(dataset, "Postures.csv"), na_values="?")
            df = df.dropna(axis=1)
            # Skip the first row which contains an "empty" measruments. Its the only one with class 0
            df = df.iloc[1:]
            df.pop("User")
            label = df.pop("Class")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "anura":
            df = pd.read_csv(os.path.join(dataset, "Frogs_MFCCs.csv"), header=0)
            df = df.dropna(axis=1)
            df.pop("RecordID")
            df.pop("Family")
            df.pop("Genus")
            label = df.pop("Species")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "shill":
            df = pd.read_csv(os.path.join(dataset, "Shill Bidding Dataset.csv"), header=0)
            df = df.dropna(axis=1)
            df.pop("Record_ID")
            df.pop("Auction_ID")
            df.pop("Bidder_ID")
            label = df.pop("Class")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "cardiotocography":
            df = pd.read_csv(os.path.join(dataset, "data.csv"), header=0)
            df = df.dropna(axis=1)
            label = df.pop("Class")
            le = LabelEncoder()
            Y = le.fit_transform(label)
            X = df.values
        elif dataset == "nursery":
            df = pd.read_csv(os.path.join(dataset, "nursery.data"), header=None)
            df = df.dropna(axis=1)
            df = df.iloc[:,:-1]
            label = df.iloc[:,-1]
            # From the documentation there should be 5 classes not_recom, recommend, very_recom, priority, spec_prior. But the data we got only seems to contain 3 classes
            # print(label.unique())
            #label.replace("recommend", "very_recom", inplace=True)
            # print(label.unique())
            le = LabelEncoder()
            Y = le.fit_transform(label)
            df = pd.get_dummies(df)
            X = df.values
        elif dataset == "susy":
            df = pd.read_csv(os.path.join(dataset, "SUSY.csv.gz"),  compression='gzip', header=None)
            Y = df.pop(0).values
            X = df.values
        else:
            exit(1)

        np.random.seed(12345)

        # scaler = MinMaxScaler()
        # X = scaler.fit_transform(X)
        kf = StratifiedKFold(n_splits=args.xval, random_state=12345, shuffle=True)
        idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X, Y)], dtype=object)

        from collections import Counter
        print("Data: ", X.shape, " ", X[0:2,:])
        print("Labels: ", Y.shape, " ", Counter(Y))
        # exit(1)
        # print("Data: ", X.shape)
        # print("Labels: ", Y.shape, " ", set(Y))
        # print("")
        #continue

        for h in args.height:
            trainidx = [itrain for itrain, _ in idx]
            testidx = [itest for _, itest in idx]

            experiment_cfg = {
                "dataset":dataset,
                "X":X,
                "Y":Y,
                "trainidx":trainidx,
                "testidx":testidx,
                "repetitions":args.xval,
                "seed":12345,
            }

            models.append(
                {
                    "model":"RandomForestClassifier",
                    "model_params":{
                        "n_estimators":args.n_estimators, 
                        "bootstrap" : True, 
                        "max_depth" : h, 
                        "n_jobs" : 32
                    },
                    **experiment_cfg
                }
            )

            for K in args.n_prune:
                models.append(
                    {
                        "model":"RandomForestClassifier",
                        "model_params":{
                            "n_estimators":K, 
                            "bootstrap" : True, 
                            "max_depth" : h, 
                            "n_jobs" : 32
                        },
                        **experiment_cfg
                    }
                )

            for loss in ["mse"]:
                for update_leaves in [True]: #True
                    for reg in [0]:
                    #for reg in [1e-5,5e-5,5e-6,1e-6,0]:
                        for K in args.n_prune:
                            models.append(
                                {
                                    "model":"ProxForestClassifier",
                                    "base_params" : {
                                        "n_estimators":args.n_estimators, 
                                        "max_depth" : h, 
                                        "splitter": "random"
                                    },
                                    "prox_params" : {
                                        "ensemble_regularizer":"hard-L0",
                                        "l_ensemble_reg":K,
                                        "l_tree_reg":reg,
                                        "batch_size" : 64,
                                        "epochs": 20,
                                        "step_size": 1e-2, 
                                        "verbose":True,
                                        "loss":loss,
                                        "update_leaves":update_leaves
                                    },
                                    "frac_pruning":0.5,
                                    **experiment_cfg
                                }
                            )


    random.shuffle(models)

    run_experiments(basecfg, models)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
    parser.add_argument("--height", help="Maximum height of the trees. Corresponds to sci-kit learns max_depth parameter. Can be a list of arguments for multiple experiments. Important: Values <= 0 are interpreted as `None` (unlimited tree depth)", nargs='+', type=int)
    parser.add_argument("-d", "--dataset", help="Dataset used for experiments",type=str, default=["wine-quality"], nargs='+')
    parser.add_argument("-n", "--n_estimators", help="Number of estimators trained for the base learner.", type=int, default=64)
    parser.add_argument("-K", "--n_prune", help="Size of the pruned ensemble. Can be a list for multiple experiments.",nargs='+', type=int, default=[32])
    parser.add_argument("-x", "--xval", help="Number of X-val runs",type=int, default=5)
    parser.add_argument("-t", "--timeout", help="Maximum number of seconds per run. If the runtime exceeds the provided value, stop execution",type=int, default=6000)
    args = parser.parse_args()

    args = parser.parse_args()

    main(args)