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
        model_ctor = NotPruningPruner
    elif cfg["model"] == "RandomPruningClassifier":
        model_ctor = RandomPruningClassifier
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
    model.prune(X, Y, estimators)
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
    scores["n_nodes"] = sum( [ est.tree_.node_count for est in from_fit.estimators_] )
    scores["n_estimators"] = len(from_fit.estimators_)
    return scores


parser = argparse.ArgumentParser()
parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
parser.add_argument("-b", "--base", help="Base learner ued for experiments. Can be {{RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, HeterogenousForest}}",type=str, default="ExtraTreesClassifier")
parser.add_argument("--height", help="Maximum height of the trees. Corresponds to sci-kit learns max_depth parameter. Can be a list of arguments for multiple experiments. Important: Values <= 0 are interpreted as `None` (unlimited tree depth)", nargs='+', type=int)
parser.add_argument("-d", "--dataset", help="Dataset used for experiments",type=str, default="bank", nargs='+')
parser.add_argument("-n", "--n_estimators", help="Number of estimators trained for the base learner.", type=int, default=64)
parser.add_argument("-T", "--n_prune", help="Size of the pruned ensemble. Can be a list for multiple experiments.",nargs='+', type=int, default=[32])
parser.add_argument("-x", "--xval", help="Number of X-val runs",type=int, default=5)
parser.add_argument("-p", "--use_prune", help="Use a train / prune / test split. If false, the training data is also used for pruning", action="store_true", default=True)
parser.add_argument("-t", "--timeout", help="Maximum number of seconds per run. If the runtime exceeds the provided value, stop execution",type=int, default=3600)
args = parser.parse_args()

if not args.base in ["RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier", "HeterogenousForest"]:
    print("Choose one of the following base learner: {{RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, HeterogenousForest}}")
    exit(1)

# if not args.dataset in ["magic", "covtype", "letter", "bank", "adult", "shuttle", "dry-beans", "eeg", "elec", "wine-quality", "thyroid", "pen-digits", "mushroom", "spambase", "satimage", "japanese-vowels", "gas-drift", "connect", "mozilla"]:
#     print("You choose {} as a dataset. Please choose one of the following datasets: {{magic, covtype, letter, bank, adult, shuttle, dry-beans, eeg, elec, wine-quality, thyroid, pen-digits, mushroom, spambase, satimage, japanese-vowels, gas-drift, connect, mozilla}}".format(args.dataset))
#     exit(1)

if args.height is None:
    args.height = [None]
else:
    args.height = [None if h <= 0 else h for h in args.height]

if args.n_prune is None:
    args.n_prune = [32]

if len(args.dataset) == 1:
    outpath = args.dataset
    args.dataset = [args.dataset]
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
    print("Loading {}".forma(dataset))

    if args.dataset == "magic":
        #df = pd.read_csv(os.path.join(args.dataset, "magic04.data"))
        df = pd.read_csv(os.path.join(args.dataset, "magic04.data"))
        X = df.values[:,:-1].astype(np.float64)
        Y = df.values[:,-1]
        Y = np.array([0 if y == 'g' else 1 for y in Y])
    elif args.dataset == "covtype":
        df = pd.read_csv(os.path.join(args.dataset, "covtype.data"), header=None)
        X = df.values[:,:-1].astype(np.float64)
        Y = df.values[:,-1]
        Y = Y - min(Y)
    elif args.dataset == "letter":
        df = pd.read_csv(os.path.join(args.dataset, "letter-recognition.data"), header=None)
        X = df.values[:,1:].astype(np.float64)
        Y = df.values[:,0]
        Y = np.array( [ord(y) - 65 for y in Y] )
    elif args.dataset == "adult":
        col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
            "hours-per-week", "native-country", "label"
        ]
        df = pd.read_csv(os.path.join(args.dataset, "adult.data"), header=None, names=col_names)
        df = df.dropna()
        label = df.pop("label")
        Y = np.array([0 if l == " <=50K" else 1 for l in label])
        df = pd.get_dummies(df)
        X = df.values
    elif args.dataset == "bank":
        df = pd.read_csv(os.path.join(args.dataset, "bank-full.csv"), header=0, delimiter=";")
        df = df.dropna()
        label = df.pop("y")
        Y = np.array([0 if l == "no" else 1 for l in label])
        df = pd.get_dummies(df)
        X = df.values
    elif args.dataset == "shuttle":
        df = pd.read_csv(os.path.join(args.dataset, "shuttle.tst"), delimiter=" ")
        Y = df.values[:,-1]
        Y = Y - min(Y)
        Y = np.array( [1 if y > 0 else 0 for y in Y] )
        X = df.values[:,:-1]
    elif args.dataset == "dry-beans":
        df = pd.read_excel(os.path.join(args.dataset,"DryBeanDataset","Dry_Bean_Dataset.xlsx"), header = 0)
        df = df.dropna()
        label = df.pop("Class")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    elif args.dataset == "spambase":
        df = pd.read_csv(os.path.join(args.dataset,"spambase.csv"), header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("class")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    elif args.dataset == "satimage":
        df = pd.read_csv(os.path.join(args.dataset,"satimage.csv"), header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("class")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    elif args.dataset == "connect":
        df = pd.read_csv(os.path.join(args.dataset,"connect.csv"), header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("class")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    elif args.dataset == "mozilla":
        df = pd.read_csv(os.path.join(args.dataset,"mozilla.csv"), header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("state")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    elif args.dataset == "mushroom":
        df = pd.read_csv(os.path.join(args.dataset,"mushroom.csv"), header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("class")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = pd.get_dummies(df).values
    elif args.dataset in ["eeg", "elec"]:
        if args.dataset == "eeg":
            data, meta = loadarff(os.path.join("eeg", "EEG Eye State.arff"))
        else:
            data, meta = loadarff(os.path.join("elec", "elecNormNew.arff"))

        Xdict = {}
        for cname, ctype in zip(meta.names(), meta.types()):
            # Get the label attribute for the specific dataset:
            #   eeg: eyeDetection
            #   elec: class
            if cname in ["eyeDetection", "class"]:
                enc = LabelEncoder()
                Xdict["label"] = enc.fit_transform(data[cname])
            else:
                Xdict[cname] = data[cname]
        df = pd.DataFrame(Xdict)
        df = pd.get_dummies(df)
        Y = df["label"].values.astype(np.int32)
        df = df.drop("label", axis=1)

        X = df.values.astype(np.float64)
    elif args.dataset == "wine-quality":
        df = pd.read_csv(os.path.join(args.dataset,"data.csv"), header = 0, delimiter=";")
        df = df.dropna()
        label = df.pop("quality")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    elif args.dataset == "thyroid":
        df = pd.read_csv(os.path.join(args.dataset,"ann-train.data"), header = None, delimiter=" ")
        # For some reason there are two whitespaces at the end of each line
        label = df.values[:,-3]
        X = df.values[:,:-3]
        le = LabelEncoder()
        Y = le.fit_transform(label)
    elif args.dataset == "pen-digits":
        df = pd.read_csv(os.path.join(args.dataset,"data.txt"), header = None, delimiter=",")
        label = df.values[:,-1]
        X = df.values[:,:-1]
        le = LabelEncoder()
        Y = le.fit_transform(label)
    elif args.dataset == "japanese-vowels":
        df = pd.read_csv(os.path.join(args.dataset,"japanese-vowels.csv"), header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("speaker")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    elif args.dataset == "gas-drift":
        df = pd.read_csv(os.path.join(args.dataset,"gas-drift.csv"), header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("Class")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    else:
        exit(1)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    kf = StratifiedKFold(n_splits=args.xval, random_state=12345, shuffle=True)
    idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X, Y)], dtype=object)

    from collections import Counter
    print("Data: ", X.shape, " ", X[0:2,:])
    print("Labels: ", Y.shape, " ", Counter(Y))
    # print("Labels: ", Y.shape, " ", set(Y))

    # HeterogenousForest requires a list of heights so pack it into another list
    if args.base == "HeterogenousForest":
        heights = [args.height]
    else:
        heights = args.height

    for h in heights:
        print("Training initial {} with h = {}".format(args.base, h))

        trainidx = []
        testidx = []
        pruneidx = []

        estimators = []
        for itrain, itest in idx:
            if args.base == "RandomForestClassifier":
                base_model = RandomForestClassifier(n_estimators = args.n_estimators, bootstrap = True, max_depth = h, n_jobs = args.n_jobs)
            elif args.base == "ExtraTreesClassifier":
                base_model = ExtraTreesClassifier(n_estimators = args.n_estimators, bootstrap = True, max_depth = h, n_jobs = args.n_jobs)
            elif args.base == "BaggingClassifier":
                # This pretty much fits a RF with max_features = None / 1.0
                base_model = BaggingClassifier(n_estimators = args.n_estimators, bootstrap = True, max_depth = h, n_jobs = args.n_jobs)
            else:
                base_model = HeterogenousForest(base=RandomForestClassifier, n_estimators = args.n_estimators, depth = h, n_jobs = args.n_jobs)
                
            if args.use_prune:
                XTrain, _, YTrain, _, tmp_train, tmp_prune = train_test_split(X[itrain], Y[itrain], itrain, test_size = 0.33)
                trainidx.append(tmp_train)
                pruneidx.append(tmp_prune)
            else:
                trainidx.append(itrain)
                pruneidx.append(itrain)
            
            testidx.append(itest)
            base_model.fit(XTrain, YTrain)
            estimators.append(copy.deepcopy(base_model.estimators_))

        experiment_cfg = {
            "dataset":dataset,
            "X":X,
            "Y":Y,
            "trainidx":trainidx,
            "testidx":testidx,
            "pruneidx":pruneidx,
            "repetitions":args.xval,
            "seed":12345,
            "estimators":estimators,
            "height":h,
            "base":args.base
        }

        np.random.seed(experiment_cfg["seed"])

        models.append(
            {
                "model":args.base,
                "model_params":{
                    "n_estimators":args.n_estimators
                },
                **experiment_cfg
            }
        )

        for K in args.n_prune:
            for m in ["individual_margin_diversity", "individual_contribution", "individual_error", "individual_kappa_statistic", "reduced_error", "complementariness", "margin_distance", "combined", "drep"]:
                models.append(
                    {
                        "model":m,
                        "model_params":{
                            "n_estimators":K
                        },
                        **experiment_cfg
                    }
                )

        for loss in ["mse", "cross-entropy"]:
            for update_leaves in [False, True]:
                for K in args.n_prune:
                    for reg in [1e-3,1e-4,1e-5,1e-6,0]:
                        models.append(
                            {
                                "model":"ProxPruningClassifier",
                                "model_params":{
                                    "ensemble_regularizer":"hard-L1",
                                    "l_ensemble_reg":K,
                                    "l_tree_reg":reg,
                                    "batch_size" : 32,
                                    "epochs": 20,
                                    "step_size": 1e-2,
                                    "verbose":False,
                                    "loss":loss,
                                    "update_leaves":update_leaves
                                },
                                **experiment_cfg
                            }
                        )
    

random.shuffle(models)

run_experiments(basecfg, models)
