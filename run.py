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
    scores["n_nodes"] = sum( [ est.tree_.node_count for est in from_fit.estimators_] )
    scores["n_estimators"] = len(from_fit.estimators_)
    return scores


parser = argparse.ArgumentParser()
parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
parser.add_argument("-b", "--base", help="Base learner ued for experiments. Can be {{RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, HeterogenousForest}}",type=str, nargs='+', default=["RandomForestClassifier"])
parser.add_argument("--height", help="Maximum height of the trees. Corresponds to sci-kit learns max_depth parameter. Can be a list of arguments for multiple experiments. Important: Values <= 0 are interpreted as `None` (unlimited tree depth)", nargs='+', type=int)
parser.add_argument("-d", "--dataset", help="Dataset used for experiments",type=str, default=["wine-quality"], nargs='+')
parser.add_argument("-n", "--n_estimators", help="Number of estimators trained for the base learner.", type=int, default=64)
parser.add_argument("-T", "--n_prune", help="Size of the pruned ensemble. Can be a list for multiple experiments.",nargs='+', type=int, default=[32])
parser.add_argument("-x", "--xval", help="Number of X-val runs",type=int, default=5)
parser.add_argument("-p", "--use_prune", help="Use a train / prune / test split. If false, the training data is also used for pruning", action="store_true", default=False)
parser.add_argument("-t", "--timeout", help="Maximum number of seconds per run. If the runtime exceeds the provided value, stop execution",type=int, default=6000)
args = parser.parse_args()

# if not args.base in ["RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier", "HeterogenousForest"]:
#     print("Choose one of the following base learner: {{RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, HeterogenousForest}}")
#     exit(1)

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
    else:
        exit(1)

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

    # HeterogenousForest requires a list of heights so pack it into another list
    for base in args.base:
        # if base == "HeterogenousForest":
        #     heights = [args.height]
        # else:
        #     heights = args.height

        for h in args.height:
            print("Training initial {} with h = {}".format(base, h))

            trainidx = []
            testidx = []
            pruneidx = []

            estimators = []
            classes = []
            for itrain, itest in idx:
                if base == "RandomForestClassifier":
                    base_model = RandomForestClassifier(n_estimators = args.n_estimators, bootstrap = True, max_depth = h, n_jobs = args.n_jobs)
                elif base == "ExtraTreesClassifier":
                    base_model = ExtraTreesClassifier(n_estimators = args.n_estimators, bootstrap = True, max_depth = h, n_jobs = args.n_jobs)
                elif base == "BaggingClassifier":
                    # This pretty much fits a RF with max_features = None / 1.0
                    base_model = BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth = h),n_estimators = args.n_estimators, bootstrap = True, n_jobs = args.n_jobs)
                else:
                    base_model = HeterogenousForest(base=DecisionTreeClassifier, n_estimators = args.n_estimators, max_depth = h, splitter = "random", bootstrap=True, n_jobs = args.n_jobs)
                
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
                "repetitions":args.xval,
                "seed":12345,
                "estimators":estimators,
                "classes":classes,
                "n_classes":len(set(Y)),
                "height":h,
                "base":base
            }

            np.random.seed(experiment_cfg["seed"])

            models.append(
                {
                    "model":base,
                    "model_params":{
                        "n_estimators":args.n_estimators
                    },
                    **experiment_cfg
                }
            )

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

            # "combined"
            for K in args.n_prune:
                for m in ["individual_margin_diversity", "individual_contribution", "individual_error", "individual_kappa_statistic", "reduced_error", "complementariness", "margin_distance",  "RandomPruningClassifier", "reference_vector", "error_ambiguity"]:
                    models.append(
                        {
                            "model":m,
                            "model_params":{
                                "n_estimators":K
                            },
                            **experiment_cfg
                        }
                    )

            rho = [0.25,0.3,0.35,0.4,0.45,0.5]
            for K in args.n_prune:
                for r in rho:
                    models.append(
                        {
                            "model":"drep",
                            "model_params":{
                                "n_estimators":K,
                                "rho": r
                            },
                            **experiment_cfg
                        }
                    ) 

            for loss in ["mse"]:
                for update_leaves in [False, True]: #True
                    #for reg in [0]:
                    for reg in [1e-5,5e-5,5e-6,1e-6,0]:
                        for K in args.n_prune:
                        #for reg in [0]:
                            models.append(
                                {
                                    "model":"ProxPruningClassifier",
                                    "model_params":{
                                        "ensemble_regularizer":"hard-L1",
                                        "l_ensemble_reg":K,
                                        "l_tree_reg":reg,
                                        "batch_size" : 128,
                                        "epochs": 50,
                                        "step_size": 1e-2, 
                                        "verbose":False,
                                        "loss":loss,
                                        "update_leaves":update_leaves
                                    },
                                    **experiment_cfg
                                }
                            )

                        # for sr in [1.0,5e-1,1e-1,5e-2,1e-2,1e-3]:
                        #     models.append(
                        #         {
                        #             "model":"ProxPruningClassifier",
                        #             "model_params":{
                        #                 "ensemble_regularizer":"L1",
                        #                 "l_ensemble_reg":sr,
                        #                 "l_tree_reg":reg,
                        #                 "batch_size" : 32,
                        #                 "epochs": 50,
                        #                 "step_size": 1e-3,
                        #                 "verbose":False,
                        #                 "loss":loss,
                        #                 "update_leaves":update_leaves
                        #             },
                        #             **experiment_cfg
                        #         }
                        #     )
    

random.shuffle(models)

run_experiments(basecfg, models)
