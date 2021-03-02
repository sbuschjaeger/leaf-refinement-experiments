#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import random

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from experiment_runner.experiment_runner import run_experiments, Variation, generate_configs

from PyPruning.RandomPruningClassifier import RandomPruningClassifier

def get_by_name(name):
    if name == "RandomForestClassifier":
        return RandomForestClassifier
    elif name == "RandomPruningClassifier":
        return RandomPruningClassifier
    else:
        return None

def pre(cfg):
    model_ctor = get_by_name(cfg["model"])
    model_params = cfg["model_params"]

    if "out_path" in model_params and model_params["out_path"] is not None:
        model_params["out_path"] = cfg["out_path"]

    if "base" in cfg:
        base_ctor = get_by_name(cfg["base"]["model"])
        base_params = cfg["base"]["model_params"]
        base = base_ctor(**base_params)
        i = cfg["run_id"]
        itrain, _ = cfg["idx"][i]
        X, Y = cfg["X"],cfg["Y"]
        XTrain, XPrune, YTrain, YPrune = train_test_split(X, Y, test_size = 0.25)
        base.fit(XTrain, YTrain)
        estimators = base.estimators_
    else:
        estimators, XPrune, YPrune = None, None, None

    model = model_ctor(**model_params)
    return (model, estimators, XPrune, YPrune)

def fit(cfg, from_pre):
    model, estimators, XPrune, YPrune = from_pre

    if estimators is None:
        i = cfg["run_id"]
        itrain, _ = cfg["idx"][i]
        X, Y = cfg["X"],cfg["Y"]

        model.fit(X[itrain], Y[itrain])
    else:
        model.prune(XPrune, YPrune, estimators)

    return model

def post(cfg, from_fit):
    scores = {}
    i = cfg["run_id"]
    _, itest = cfg["idx"][i]
    X, Y = cfg["X"],cfg["Y"]
    pred = from_fit.predict_proba(cfg["X"][itest])

    scores["accuracy"] = 100.0 * accuracy_score(cfg["Y"][itest], pred.argmax(axis=1))
    scores["n_nodes"] = sum( [ est.tree_.node_count if w != 0 else 0 for w, est in zip(from_fit.weights_, from_fit.estimators_)] )
    return scores


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--single", help="Run via single thread",action="store_true", default=False)
parser.add_argument("-m", "--multi", help="Run via multiprocessing pool",action="store_true", default=False)
parser.add_argument("-d", "--dataset", help="Dataset used for experiments",type=str, default="magic")
#parser.add_argument("-c", "--n_configs", help="Number of configs evaluated per method",type=int, default=50)
parser.add_argument("-x", "--xval", help="Number of X-val runs",type=int, default=5)
parser.add_argument("-t", "--timeout", help="Maximum number of seconds per run. If the runtime exceeds the provided value, stop execution",type=int, default=1800)
args = parser.parse_args()

if not args.single and not args.multi:
    print("No processing mode found, defaulting to `single` processing.")
    args.single = True

if not args.dataset in ["magic"]:
    print("Choose one of the following datasets: {{magic}}")
    exit(1)

if args.single:
    basecfg = {
        "out_path":os.path.join(args.dataset, "results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "single",
        "verbose":True,
        "timeout":args.timeout
    }
elif args.multi:
    basecfg = {
        "out_path":os.path.join(args.dataset, "results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "multiprocessing",
        "num_cpus":8,
        "verbose":True,
        "timeout":args.timeout
    }
else:
    exit(1)

print("Loading data")
if args.dataset == "magic":
    df = pd.read_csv(os.path.join(args.dataset, "magic04.data"))
    X = df.values[:,:-1].astype(np.float64)
    Y = df.values[:,-1]
    Y = np.array([0 if y == 'g' else 1 for y in Y])


scaler = MinMaxScaler()
X = scaler.fit_transform(X)
kf = KFold(n_splits=args.xval, random_state=12345, shuffle=True)
idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X)], dtype=object)

experiment_cfg = {
    "X":X,
    "Y":Y,
    "idx":idx,
    "repetitions":args.xval,
    "verbose":True,
    "seed":12345
}

np.random.seed(experiment_cfg["seed"])

models = []
# models.append(
#     {
#         "model":"RandomForestClassifier",
#         "model_params":{
#             "n_estimators":1024,
#             "bootstrap":True,
#             "max_depth":None
#         },
#         **experiment_cfg
#     }
# )

#, 32, 64, 128
for T in [16]:
    models.append(
        {
            "model":"RandomPruningClassifier",
            "model_params":{
                "n_estimators":T
            },
            "base": {
                "model":"RandomForestClassifier",
                "model_params":{
                    "n_estimators":128,
                    "bootstrap":True,
                    "max_depth":None
                }
            },
            **experiment_cfg
        }
    )
random.shuffle(models)

run_experiments(basecfg, models)
