# %%
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize 
import os
import json 
from IPython.display import display, HTML

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return json_normalize(data)

def nice_name(row):
    if row["model"] == "RandomForestClassifier":
        return "RandomForestClassifier d {}".format(row["height"])
    elif row["model"] == "ExtraTreesClassifier d {}".format(row["height"]):
        return "ExtraTreesClassifier"
    elif row["model"] == "HeterogenousRFClassifier":
        return "HeterogenousRFClassifier d {}".format(row["height"])
    elif row["model"] == "ProxPruningClassifier":
        # if row["base.model"] == "RandomForestClassifier":
        #     base_type = "RF"
        # elif row["base.model"] == "ExtraTreesClassifier":
        #     base_type = "ET"
        # else:
        #     base_type = "HF"
        return "HEP-{}-{}-{} λ1={},λ2={},UL {},d {}".format("ET", row["model_params.ensemble_regularizer"], row["model_params.loss"], row["model_params.l_ensemble_reg"], row["model_params.l_tree_reg"], row["model_params.update_leaves"],row["height"])
    else:
        return "{} with {}, d {}".format(row["model"], "ET",row["height"])
        # if row["base.model"] == "RandomForestClassifier":
        #     return "{} with {}".format(row["model"], "RF")
        # elif row["base.model"] == "ExtraTreesClassifier":
        #     return "{} with {}".format(row["model"], "ET")
        # else:
        #     return "{} with {}".format(row["model"], "HF")


#dataset = "adult"
#dataset = "bank"
dataset = "connect"
#dataset = "covtype"
#dataset = "dry-beans"
#dataset = "eeg"
#dataset = "elec"
#dataset = "gas-drift"
#dataset = "japanese-vowels"
#dataset = "letter"
#dataset = "magic"
#dataset = "mozilla"
#dataset = "mushroom"
#dataset = "pen-digits"
#dataset = "satimage"
#dataset = "shuttle"
#dataset = "spambase"
#dataset = "thyroid"
#dataset = "wine-quality"

dataset = os.path.join(dataset, "results")
all_subdirs = [os.path.join(dataset,d) for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
#print(all_subdirs)
latest_folder = max(all_subdirs, key=os.path.getmtime)
#latest_folder = "eeg/results/19-02-2021-13:29:39"

print("Reading {}".format(os.path.join(latest_folder, "results.jsonl")))
df = read_jsonl(os.path.join(latest_folder, "results.jsonl"))
df["nice_name"] = df.apply(nice_name, axis=1)

df["accuracy"] = df["scores.mean_accuracy"]
df["n_nodes"] = df["scores.mean_n_nodes"]
df["fit_time"] = df["scores.mean_fit_time"]
df["n_estimators"] = df["scores.mean_n_estimators"]
#df["roc_auc"] = df["scores.mean_roc_auc"]

# Lets assume a native implementation with
#  - unsigned int left, right
#  - bool is_leaf
#  - unsinged int feature_idx
#  - float threshold
#  - float pred[n_classes]
#   ==> 4+4+1+4+4+4*n_classes = 17 + 4*n_classes
if dataset == "letter":
    n_classes = 26
elif dataset == "thyroid":
    n_classes = 3
elif dataset == "covtype":
    n_classes = 7
elif dataset == "wine-quality":
    n_classes = 6
elif dataset == "pen-digits":
    n_classes = 10
else:
    n_classes = 2
df["KB"] = df["scores.mean_n_nodes"] * (17 + 4*n_classes)  / 1024.0

df = df.round(decimals = 3)
tabledf = df[["nice_name", "accuracy", "n_nodes", "fit_time", "n_estimators", "KB","height"]]
#tabledf = tabledf.loc[tabledf["n_estimators"] < 100]
# tabledf = tabledf.loc[tabledf["KB"] < 512]
#tabledf = tabledf.loc[tabledf["height"] > 10]

tabledf = tabledf.sort_values(by=['accuracy'], ascending = False)
print("Processed {} experiments".format(len(tabledf)))
display(HTML(tabledf.to_html()))

print("Best configuration per group")
# print("Experiments per group")
# print(tabledf.groupby(['nice_name'])['accuracy'].count())
# idx = tabledf.groupby(['nice_name'])['accuracy'].transform(max) == tabledf['accuracy']
# shortdf = tabledf[idx]

# print("Best configuration per group")
# display(HTML(shortdf.to_html()))

# %%
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

fdf = df.loc[df["nice_name"] == "PE"]
fdf = fdf[["nice_name", "mean_accuracy","model_params.loss", "model_params.batch_size", "model_params.max_depth", "model_params.step_size", "model_params.l_ensemble_reg", "model_params.update_trees"]]
fdf = fdf.sort_values(by=['mean_accuracy'], ascending = False)
display(HTML(fdf.to_html()))

idx = [256, 462, 492]

for i in idx:
    dff = df.iloc[i]
    metrics = np.load(os.path.join(dff["out_path"], "epoch_0.npy"), allow_pickle=True).item()
    avg_acc = []
    sum_acc = 0.0
    for a in metrics["accuracy"]:
        sum_acc += a
        avg_acc.append( sum_acc / (len(avg_acc) + 1))


    plt.plot(range(0,len(metrics["accuracy"])), avg_acc, label=str(i))

plt.legend(loc="lower right")
plt.show()

# %%
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


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
    # # Return ids of scenarios on pareto front
    # return population_ids[pareto_front]


for name, group in df.groupby(["nice_name"]):
    pdf = get_pareto(group, ["mean_accuracy", "mean_params"])
    pdf = pdf[["nice_name", "mean_accuracy", "mean_params", "scores.mean_fit_time"]]
    print(pdf)
    pdf = pdf.sort_values(by=['mean_accuracy'], ascending = False)
    plt.plot(pdf["mean_params"].values, pdf["mean_accuracy"], linestyle='solid', label=name)

plt.legend(loc="lower right")

#%%

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=9
paired = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'] 
colors = {}
for m,c in zip(df["nice_name"].values, paired):
    colors[m] = c

fig = make_subplots(rows=3, cols=1, subplot_titles=["Covtype"], horizontal_spacing = 0.03, vertical_spacing = 0.02)

for tdf, m in zip(traindfs, df["nice_name"].values):
    fig = fig.add_trace(go.Scatter(x=tdf["total_item_cnt"], y = tdf["item_loss"], mode="lines", name = m, marker=dict(color = colors[m])), row = 1, col = 1)
    fig = fig.add_trace(go.Scatter(x=tdf["total_item_cnt"], y = tdf["item_accuracy"], mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 2, col = 1)
    fig = fig.add_trace(go.Scatter(x=tdf["total_item_cnt"], y = tdf["item_num_parameters"], mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 3, col = 1)

fig.update_xaxes(title_text="Number of items", row=3, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="Loss", row=1, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="Accuracy", row=2, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="Num of trainable parameters", row=3, col=1, title_font = {"size": 16})

fig.update_layout(
    template="simple_white",
    legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="left",x=0.15),
    margin={'l': 5, 'r': 20, 't': 20, 'b': 5},
    height=900, width=1100
)
fig.show()


#%%
sub_experiments = [os.path.join(experiment_path,d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]

if len(sub_experiments) == 0:
    sub_experiments = [os.path.join(experiment_path, "epoch_0.npy")]

accuracies = []
losses = []
num_nodes = []
times = []
total_item_cnt = None
for experiment in sub_experiments:
    print("Reading {}".format(experiment))
    tdf = read_jsonl(experiment)
    losses.append(tdf["item_loss"].values)
    accuracies.append(tdf["item_accuracy"].values)
    num_nodes.append(tdf["item_num_parameters"].values)
    times.append(tdf["item_time"].values)
    if total_item_cnt is None:
        total_item_cnt = tdf["total_item_cnt"]

    
d = {
    "total_item_cnt":total_item_cnt,
    "item_loss":np.mean(losses, axis=0),
    "item_accuracy":np.mean(accuracies, axis=0),
    "item_num_parameters":np.mean(num_nodes, axis=0),
    "item_time":np.mean(times,axis=0)
}
