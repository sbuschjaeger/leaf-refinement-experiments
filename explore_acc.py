# %%
from typing import final
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize 
import os
import json 
from IPython.display import display, HTML

def read_jsonl(path):
    '''
    Reads the given *.jsonl file and normalizes it to produce a flattend pandas frame.
    '''
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return json_normalize(data)

def memory(row):
    '''
    Compute the memory consumption for the model in the current row in KB.
    Lets assume a native implementation with
        - unsigned int left, right
        - bool is_leaf
        - unsinged int feature_idx
        - float threshold
        - float pred[n_classes]
    ==> 4+4+1+4+4+4*n_classes = 17 + 4*n_classes
    '''
    if row["dataset"] == "anura":
        n_classes = 10
    elif row["dataset"] == "avila":
        n_classes = 11
    elif row["dataset"] == "cardiotocography":
        n_classes = 10
    elif row["dataset"] == "connect":
        n_classes = 3
    elif row["dataset"] == "covtype":
        n_classes = 7
    elif row["dataset"] == "dry-beans":
        n_classes = 7
    elif row["dataset"] == "gas-drift":
        n_classes = 5
    elif row["dataset"] == "japanese-vowels":
        n_classes = 9
    if row["dataset"] == "letter":
        n_classes = 26
    elif row["dataset"] == "mnist":
        n_classes = 10
    elif row["dataset"] == "nursery":
        n_classes = 3
    elif row["dataset"] == "pen-digits":
        n_classes = 10
    elif row["dataset"] == "postures":
        n_classes = 5
    elif row["dataset"] == "satimage":
        n_classes = 6
    elif row["dataset"] == "thyroid":
        n_classes = 3
    elif row["dataset"] == "weight-lifting":
        n_classes = 5
    elif row["dataset"] == "wine-quality":
        n_classes = 7
    else:
        n_classes = 2

    return row["scores.mean_n_nodes"] * (17 + 4*n_classes)  / 1024.0

def nice_name(row, split_hep = False, split_lambda = False):
    '''
    Sanitize names for later plotting. 
        split_hep: If true we split HEP into HEP and HEP-LR
        split_lambda: If true, we split HEP (HEP-LR) into the different lambda values
    '''
    if row["model"] == "ExtraTreesClassifier":
        return "ET"
    elif row["model"] == "RandomForestClassifier":
        return "RF-{}".format(row["model_params.n_estimators"])
    elif row["model"] == "ProxPruningClassifier":
        if split_hep:
            if row["model_params.update_leaves"]:
                if split_lambda:
                    return "HEP-LR λ = {}".format(row["model_params.l_tree_reg"])
                else:
                    return "HEP-LR"
            else:
                if split_lambda:
                    return "HEP λ = {}".format(row["model_params.l_tree_reg"])
                else:
                    return "HEP"
        else:
            return "HEP"
    elif row["model"] == "RandomPruningClassifier":
        return "rand."
    elif row["model"] == "complementariness":
        return "comp."
    elif row["model"] == "drep":
        return "DREP"
    elif row["model"] == "individual_contribution":
        return "IC"
    elif row["model"] == "individual_error":
        return "IE"
    elif row["model"] == "individual_kappa_statistic":
        return "IKS"
    elif row["model"] == "individual_margin_diversity":
        return "IMD"
    elif row["model"] == "margin_distance":
        return "MD"
    elif row["model"] == "reduced_error":
        return "RE"
    elif row["model"] == "HeterogenousForest":
        return "HF"
    elif row["model"] == "reference_vector":
        return "RV"
    elif row["model"] == "error_ambiguity":
        return "EA"
    elif row["model"] == "largest_mean_distance":
        return "LMD"
    elif row["model"] == "cluster_accuracy":
        return "CA"
    elif row["model"] == "cluster_centroids":
        return "CC"
    elif row["model"] == "combined_error":
        return "CE"
    elif row["model"] == "combined":
        return "comb,"
    else:
        return row["model"]

def highlight(s):
    '''
    Nice styling of inline tables. This highlights the best method ( = best rank), smallest model (= smallest memory) and best accuracy.
    This helps a lot when reviewing the results. Probably has only effect if Jupyter / VSCode is used.
    '''
    accs = []
    kbs = []
    ranks = []
    for i in range(0, len(s), 3):
        accs.append(s[i])
        kbs.append(s[i+1])
        ranks.append(s[i+2])

    max_acc = np.nanmax(accs)
    min_kb = np.nanmin(kbs)
    min_rank = np.nanmin(ranks)

    style = []
    for acc, kb, rank in zip(accs, kbs, ranks):
        if acc == max_acc:
            style.append('background-color: blue; text-align: left')
        else:
            style.append('')
        
        if kb == min_kb:
            style.append('background-color: green')
        else:
            style.append('')

        if rank == min_rank:
            style.append('background-color: red')
        else:
            style.append('')
    return style

# %%

import matplotlib.pyplot as plt
from itertools import cycle
import scipy.stats as ss
import scikit_posthocs as sp
from functools import partial

# Set the theme for plotting and how to process the results.
#   - plot: if plot is true the plots from the paper are created, displayed and stored in pdf files. If this is false, tables are exported as HTML
#   - split_hep: If true, the HEP is splitted into HEP and HEP-LR (as done for Q2 in the paper) 
#   - split_lambda: If true, the HEP is splitted into the various lambda values used during the experiments (as done for Q3 in the paper) 

plt.style.use('seaborn-whitegrid')
plot = False
split_hep = True
split_lambda = False

# Select the dataset which should be plotted and navigate to the youngest folder
# If you have another folder-structure you can comment out this code and simply set latest_folder to the correct path
# If you ran experiments on multiple datasets the corresponding folder is called "multi" 
dataset = "susy"
dataset = os.path.join(dataset, "results")
all_subdirs = [os.path.join(dataset,d) for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
latest_folder = max(all_subdirs, key=os.path.getmtime)
print("Reading {}".format(os.path.join(latest_folder, "results.jsonl")))

# Read the file 
df = read_jsonl(os.path.join(latest_folder, "results.jsonl"))
print("Reading done")

# Compute nicer model names and the memory consumption
df["model"] = df.apply(partial(nice_name, split_hep = split_hep, split_lambda = split_lambda), axis=1)
df["KB"] = df.apply(memory, axis=1)

# Rename some columns for readability
df["test_accuracy"] = df["scores.mean_test_accuracy"]
df["train_accuracy"] = df["scores.mean_train_accuracy"]
df["n_nodes"] = df["scores.mean_n_nodes"]
df["fit_time"] = df["scores.mean_fit_time"]
df["n_estimators"] = df["scores.mean_n_estimators"]
df["comparisons"] = df["scores.mean_avg_comparisons_per_tree"]

dff = df.copy()
if split_lambda and plot:
    dff = dff.loc[
        dff["model"].str.contains("HEP") 
    ]

dff = dff[["model", "test_accuracy", "KB", "dataset", "comparisons", "fit_time", "train_accuracy"]]

dff.fillna(0, inplace=True)
dff = dff.loc[  dff.groupby(["dataset", "model"])["test_accuracy"].idxmax() ]
dff["rank"] = dff.groupby("dataset")["test_accuracy"].rank(axis=0, ascending=False)

# Extract the interestring parts of the table and write the raw-results to a tex file
tabledf = dff.pivot_table(["KB", "test_accuracy", "rank", "comparisons"] , ['dataset'], 'model')
tabledf = tabledf.reorder_levels([1,0], axis=1).sort_index(axis=1).reindex(["test_accuracy", "KB", "rank"], level=1, axis=1)
tabledf.fillna(0, inplace=True)

display( dff.sort_values(["dataset", "test_accuracy"], ascending=False) )

print("DONE")