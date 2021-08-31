# %%
from typing import final
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize 
import os
import json 
from IPython.display import display, HTML
from functools import partial
import matplotlib.pyplot as plt

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

def nice_name(row):
    '''
    Sanitize names for later plotting. 
    '''
    if row["model"] == "ExtraTreesClassifier":
        return "ET"
    elif row["model"] == "AdaBoostClassifier":
        return "AB"
    elif row["model"] == "RandomForestClassifier":
        return "RF"
    elif row["model"] == "ProxPruningClassifier":
        return "RF-LR"
    elif row["model"] == "RandomPruningClassifier":
        return "rand."
    elif row["model"] == "complementariness":
        return "COMP"
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

def read_data(dataset):
    # Select the dataset which should be plotted and navigate to the youngest folder
    # If you have another folder-structure you can comment out this code and simply set latest_folder to the correct path
    # If you ran experiments on multiple datasets the corresponding folder is called "multi" 

    dataset = os.path.join(dataset, "results")
    all_subdirs = [os.path.join(dataset,d) for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
    latest_folder = max(all_subdirs, key=os.path.getmtime)
    print("Reading {}".format(os.path.join(latest_folder, "results.jsonl")))

    # Read the file 
    #df = read_jsonl(os.path.join(latest_folder, "results.jsonl"))
    path = os.path.join(latest_folder, "results.jsonl")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = json_normalize(data)

    print("Read {}".format(path))

    # Compute nicer model names and the memory consumption
    df["model"] = df.apply(partial(nice_name), axis=1)
    df["KB"] = df.apply(memory, axis=1)

    # Rename some columns for readability
    df["test_accuracy"] = df["scores.mean_accuracy"]
    #df["train_accuracy"] = df["scores.mean_train_accuracy"]
    df["n_nodes"] = df["scores.mean_n_nodes"]
    df["fit_time"] = df["scores.mean_fit_time"]
    df["n_estimators"] = df["scores.mean_n_estimators"]
    df["comparisons"] = df["scores.mean_avg_comparisons_per_tree"]
    
    return df

# %%

"""
Compute the plots for the first experiments. You can set the dataset which should be plotted via the `dataset' variable. 
The `read_data' function will load the latest results from the sub-folder. The run script trains trees with 
[4096,2048, 1024, 512, 256, 128, 64] `max_leaf_nodes'. For larger `max_leaf_nodes' the plots start to 
overlap so the plot does not look nice. Hence, we decided to filter for [1024, 512, 256, 128, 64] leaf nodes. 
"""
datasets = [
    "gas-drift",
    "bank","connect",
    "eeg", "mozilla", "magic", "nomao",  
    "occupancy", "pen-digits",
    "postures", "satimage", "anura","adult"
]

show = False
for d in datasets:
    #dataset = "magic" # dataset to be plotted
    df = read_data(d)
    max_leaves_to_plot = [1024, 512, 256, 128, 64] #4096,2048 
    models_to_plot = ["RE", "RF"] # methods to be compared 
    names = []

    fig = plt.figure()

    colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    for ml,c in zip(max_leaves_to_plot, colors):
        for mt, s in zip(models_to_plot, ["solid", "dashed"]):
            dff = df.copy()
            dff = dff.loc[dff["max_leaf_nodes"] == ml]
            
            dff = dff.loc[dff["model"] == mt]
            dff.sort_values(["n_estimators"], inplace=True)
            plt.plot(dff["n_estimators"].values, dff["test_accuracy"].values, color=c, linestyle=s)
            names.append(r'{} $n_l={}$'.format(mt,ml))
        
    plt.legend(names, loc="upper right", bbox_to_anchor=(1.32, 1))
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    if show:
        plt.show()
    fig.savefig("revisited_{}.pdf".format(d), bbox_inches='tight')

# %%
"""
Compute the table for all accuracies of the different methods in the first experiment. Again, the dataset can be set via the `dataset` variable. Since this table would become to big for all experiments we now filter for `n_estimators` and `max_leaf_nodes`
"""
datasets = [
    "gas-drift",
    "bank","connect",
    "eeg", "mozilla", "magic", "nomao",  
    "occupancy", "pen-digits",
    "postures", "satimage", "anura","adult"
]
show = False

for d in datasets:
    df = read_data(d)

    max_leaves_to_table = [1024, 512, 256, 128, 64] #[16, 32, 64, 128, 2048]
    n_estimators = [8, 16, 32, 64, 128]
    models_to_table = ["RE", "RF", "IE", "IC", "COMP", "DREP", "LMD"]

    dff = df.copy()
    # dff = dff.loc[dff["dataset"] == dataset_to_plot]
    dff = dff.loc[dff["max_leaf_nodes"].isin(max_leaves_to_table)]
    dff = dff.loc[dff["model"].isin(models_to_table)]
    dff = dff.loc[dff["n_estimators"].isin(n_estimators)]
    dff.sort_values(["max_leaf_nodes","n_estimators"], inplace=True)
    dff = dff.drop_duplicates(["model","max_leaf_nodes","n_estimators"], keep="last")

    pdf = dff.pivot_table(index=["max_leaf_nodes","n_estimators"], values="test_accuracy", columns = ["model"])
    pdf.round(2).to_latex("{}_revisited.tex".format(d))
    if show:
        display(pdf.round(2))
        print(pdf.round(2).to_latex())

# %%
import scipy

def get_pareto(df, columns):
    first = df[columns[0]].values
    second = df[columns[1]].values

    # Count number of items
    population_size = len(first)
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if (first[j] >= first[i]) and (second[j] < second[i]):
            #if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    
    return df.iloc[population_ids[pareto_front]]

datasets = [
    "connect",
    "eeg",
    "postures",  
    "mnist",
    "elec", 
    "chess",
    # ----
    "avila",
    #"weight-lifting",
    #"gas-drift",
    #"occupancy",
    #"pen-digits",
    #"wine-quality",
    "japanese-vowels",
    "bank",
    "nomao",  
    "magic",
    "anura",
    "adult",
    #"dry-beans", 
    "satimage",
    "mozilla",
    "ida2016",
    #"thyroid", 
    #"weather",
    #"letter",
    #"dota2"
]

max_leaf_nodes = [64, 128, 256, 512, 1024]
show = False
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
markers = ["o", "v", "^", "<", ">", "s", "P", "X", "D"]
styles = ["-", "--", "-.", ":","-", "--", "-.", ":","-", "--", "-.", ":",]
aucs = []

for d in datasets:
    dff = read_data(d)
    #dff["test_accuracy"] = dff["test_accuracy"].round(1)

    #dff = dff.loc[dff["KB"] < 1000]
    dff = dff.loc[dff["max_leaf_nodes"].isin(max_leaf_nodes)]

    max_kb = None
    for name, group in dff.groupby(["model"]):
        if max_kb is None or group["KB"].max() > max_kb:
            max_kb = group["KB"].max()

    fig = plt.figure()
    for (name, group), marker, color, style in zip(dff.groupby(["model"]),markers, colors, styles):
        pdf = get_pareto(group, ["test_accuracy", "KB"])
        pdf = pdf[["model", "test_accuracy", "KB", "fit_time"]]
        # pdf = pdf.loc[pdf["test_accuracy"] > 86]
        #pdf = pdf.loc[pdf["KB"] > 100]
        pdf = pdf.sort_values(by=['test_accuracy'], ascending = True)
        
        x = np.append(pdf["KB"].values, [max_kb])
        y = np.append(pdf["test_accuracy"].values, [pdf["test_accuracy"].values[-1]]) / 100.0
        
        x_scatter = np.append(group["KB"].values, [max_kb])
        y_scatter = np.append(group["test_accuracy"].values,[pdf["test_accuracy"].values[-1]]) / 100.0

        plt.scatter(x_scatter,y_scatter,s = [2.5**2 for _ in x_scatter], color = color)

        plt.plot(x,y, label=name, color=color) #marker=marker
        aucs.append(
            {
                "model":name,
                #"AUC":np.trapz(y, x),
                "AUC":np.trapz(y, x) / max_kb,
                "dataset":d
            }
        )

    print("Dataset {}".format(d))
    plt.legend(loc="lower right")
    plt.xlabel("Model Size [KB]")
    plt.ylabel("Accuracy")
    fig.savefig("auc_{}.pdf".format(d), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

tabledf = pd.DataFrame(aucs)
# tabledf["ΔAUC"] = ref_auc - tabledf["AUC"]
# tabledf["AUC norm"] = tabledf["AUC"] / ref_auc
#tabledf["AUC norm"] = tabledf["AUC"] / max_kb
tabledf.sort_values(by=["dataset","AUC"], inplace = True, ascending=False)
tabledf.to_csv("aucs.csv",index=False)

tabledf.pivot_table(index=["dataset"], values=["AUC"], columns=["model"]).round(4).to_latex("aucs.tex")
#if show:
display(tabledf.pivot_table(index=["dataset"], values=["AUC"], columns=["model"]).round(4))
# %%
