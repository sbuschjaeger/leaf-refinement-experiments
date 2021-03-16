# %%
from typing import final
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

def memory(row):
    # Lets assume a native implementation with
    #  - unsigned int left, right
    #  - bool is_leaf
    #  - unsinged int feature_idx
    #  - float threshold
    #  - float pred[n_classes]
    #   ==> 4+4+1+4+4+4*n_classes = 17 + 4*n_classes
    
    if row["dataset"] == "connect":
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
    elif row["dataset"] == "pen-digits":
        n_classes = 10
    elif row["dataset"] == "satimage":
        n_classes = 6
    elif row["dataset"] == "thyroid":
        n_classes = 3
    elif row["dataset"] == "wine-quality":
        n_classes = 7
    else:
        n_classes = 2

    return row["scores.mean_n_nodes"] * (17 + 4*n_classes)  / 1024.0

# Sanitize names for later plotting
def nice_name(row):
    if row["model"] == "ExtraTreesClassifier":
        return "ET"
    elif row["model"] == "RandomForestClassifier":
        return "RF"
    elif row["model"] == "ProxPruningClassifier":
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
    else:
        return row["model"]

def highlight(s):
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

dataset = "multi"
dataset = os.path.join(dataset, "results")
all_subdirs = [os.path.join(dataset,d) for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
latest_folder = max(all_subdirs, key=os.path.getmtime)

print("Reading {}".format(os.path.join(latest_folder, "results.jsonl")))
df = read_jsonl(os.path.join(latest_folder, "results.jsonl"))
df["model"] = df.apply(nice_name, axis=1)
df["KB"] = df.apply(memory, axis=1)

df["accuracy"] = df["scores.mean_accuracy"]
df["n_nodes"] = df["scores.mean_n_nodes"]
df["fit_time"] = df["scores.mean_fit_time"]
df["n_estimators"] = df["scores.mean_n_estimators"]
print("Reading done")


# %%

import matplotlib.pyplot as plt
from itertools import cycle
import scipy.stats as ss
import scikit_posthocs as sp

plt.style.use('seaborn-whitegrid')

plot = True

rank_df = []

pval = 0.05
cliques = []

#
for kb in [16,32,64,128,256,512,None]:
#for kb in [16]:
    for b in ["RandomForestClassifier", "ExtraTreesClassifier", "HeterogenousForest", None]:
    #for b in [None]:
        dff = df.copy()
        
        # Filter for a specific base learner, but keep RF / ET / HF as well
        if b is not None:
            dff = dff.loc[
                ((dff["model"] == "RF") & (dff["base"] == "RandomForestClassifier")) |
                ((dff["model"] == "ET") & (dff["base"] == "ExtraTreesClassifier")) |
                ((dff["model"] == "HF") & (dff["base"] == "HeterogenousForest")) |
                (dff["base"] == b)
            ]
        
        # Filter for KB constraints
        if kb is not None:
            dff["accuracy"].loc[dff["KB"] > kb] = 0
        #dff = dff.fillna(value = 0)

        dff = dff[["model", "accuracy", "KB", "dataset"]]
        dff.fillna(0, inplace=True)
        dff = dff.loc[  dff.groupby(["dataset", "model"])["accuracy"].idxmax() ]
        dff["rank"] = dff.groupby("dataset")["accuracy"].rank(axis=0, ascending=False)

        if plot:
            ranks = dff.pivot_table(["rank"] , ['dataset'], 'model')
            ranks = ranks.T.fillna(ranks.max(axis=1) + 1).T
            m_order = [r[1] for r in ranks.columns]
            ranks = ranks.to_numpy().T
            #print(ranks)
            ftest = ss.friedmanchisquare(*ranks)
            # print(ftest)
            # break

            if ftest.pvalue < pval:
                tmp = sp.posthoc_wilcoxon(ranks,p_adjust="holm", zero_method="zsplit") 
                #tmp = sp.posthoc_nemenyi_friedman(ranks)
                #print(tmp)
                #break
                #, group_col='groups'
                #print("tmp ", tmp.values)
                adj = tmp.values
                adj[adj > pval] = 1.0
                adj[adj <= pval] = 0.0
                #print("KB ", kb)
                
                #print("ORDER:", m_order)
                #print(adj)
                # print("")
                cur_cliques = []
                for i, mi in enumerate(m_order):
                    clique = []
                    for j, mj in enumerate(m_order):
                        if i == j:
                            clique.append(mi)
                        elif adj[i][j] == 1.0:
                            clique.append(mj)
                    
                    cur_cliques.append(clique)

                # cur_cliques = sorted(cur_cliques, key=lambda l: len(l), reverse=True)
                # final_cliques = []

                # for i in range(len(cur_cliques)):
                #     add = False
                #     for j in range(i+1, len(cur_cliques)):
                #         f1 = cur_cliques[i]
                #         f2 = cur_cliques[j]

                #         if not f1 in final_cliques and set(f1).issubset(f2):
                #             add 
                #             final_cliques.append(f1)

                # for c in cur_cliques[1:]:
                #     add = True
                #     for 
                #     for f in final_cliques:
                #         if c in final_cliques or set(c).issubset(f):
                #             add = False
                #             break
                #     if add:
                #         final_cliques.append(c)
                # print("FC:", final_cliques)


                final_cliques = cur_cliques
                #print("kb", kb)
                #print("CUR_CLIQUES", cur_cliques)
                to_remove = []
                while True: #not removed
                    for f1 in range(len(final_cliques)):
                        for f2 in range(f1 + 1, len(final_cliques)):
                            if final_cliques[f1] == final_cliques[f2] and final_cliques[f1] in final_cliques:
                                to_remove.append(f2)
                            else:
                                if set(final_cliques[f2]).issubset(final_cliques[f1]):
                                    to_remove.append(f2)
                                if set(final_cliques[f1]).issubset(final_cliques[f2]):
                                    to_remove.append(f1)
                    if len(to_remove) == 0:
                        break
                    else:
                        # print("removing ", to_remove)
                        final_cliques = [c for i, c in enumerate(final_cliques) if i not in to_remove]
                        to_remove = []
                        # if len(final_cliques) == 1:
                        #     break
                #         print("FINAL:", final_cliques)
                #print("FINAL:", final_cliques)
                #print("")
                # cur_cliques = sorted(cur_cliques, key=lambda l: len(l), reverse=True)
                # final_cliques = [cur_cliques[0]]

                # for c in cur_cliques[1:]:
                #     add = True
                #     for f in final_cliques:
                #         if set(c).issubset(f):
                #             add = False
                #             break
                #     if add:
                #         final_cliques.append(c)
                #print("FC:", final_cliques)

                

                cliques.append(
                    {
                        "cliques": final_cliques, 
                        "base":str(b),
                        "kb":str(kb)
                    }
                )

                for m in dff["model"].unique():
                    rank_df.append(
                        {
                            "model":m,
                            "kb":str(kb),
                            "base":str(b),
                            "rank_mean":dff.loc[ dff["model"] == m ]["rank"].mean(),
                            "rank_std":dff.loc[ dff["model"] == m ]["rank"].std()
                        }
                    )

            else:
                print("No statistical difference found. Exiting")
                break
        else:
            dff = dff.pivot_table(["KB", "accuracy", "rank"] , ['dataset'], 'model')
            dff = dff.reorder_levels([1,0], axis=1).sort_index(axis=1).reindex(["accuracy", "KB", "rank"], level=1, axis=1)
            dff.fillna(0, inplace=True)
            display( dff.style.apply(highlight,axis=1) )

if plot:
    #print(rank_df)
    rank_df = pd.DataFrame(rank_df)
    clique_df = pd.DataFrame(cliques)
    markers = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd', '1', '2', '3', '4']
    for b in rank_df["base"].unique():
        groups = rank_df.loc[ rank_df["base"] == b ].groupby('model')
        
        # Plot
        fig, ax = plt.subplots()
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        for (name, group), marker in zip(groups, cycle(markers)):
            ax.plot(group["rank_mean"], group["kb"], marker=marker, linestyle='', label=name) #, alpha=0.5, , dashes=[6, 2]
        yticklabels = ["Unlimited" if kb == "None" else kb for kb in group["kb"].unique()] 
        
        ax.set_yticklabels(yticklabels)
        ax.set_ylabel("Memory constraints [KB]")
        ax.set_xlabel("Average rank")
        ax.yaxis.set_label_coords(-0.1,0.5)

        for i, kb in enumerate(clique_df.loc[ clique_df["base"] == b ]["kb"].unique()):
            cliques = clique_df.loc[ (clique_df["base"] == b) & (clique_df["kb"] == kb)]["cliques"]
            cliques = cliques.values[0]
            
            x_lines = []
            y_lines = []
            offset = 0
            x_limits = None
            for clique in cliques:
                c_ranks = []
                for c in clique:
                    c_ranks.append(
                        rank_df.loc[ 
                            (rank_df["model"] == c) &
                            (rank_df["base"] == b) &
                            (rank_df["kb"] == str(kb))
                        ]["rank_mean"].values[0]
                    )

                ybase = i
                if (min(c_ranks) != max(c_ranks)):
                    for x in x_lines:
                        if (x[1] >= min(c_ranks) and x[0] <= min(c_ranks)) or (x[0] <= max(c_ranks) and x[1] >= max(c_ranks)):
                            ybase -= 0.1

                    x_lines.append([min(c_ranks), max(c_ranks)])
                    #x_lines.append([min(c_ranks), min(c_ranks)])
                    #x_lines.append([max(c_ranks), max(c_ranks)])

                    y_lines.append([ybase - 0.25, ybase - 0.25])
                    #y_lines.append([ybase - 0.25, i - 0.1])
                    #y_lines.append([ybase - 0.25, i - 0.1])
                else:
                    x_lines.append([min(c_ranks) - 0.05, max(c_ranks) + 0.05])
                    y_lines.append([ybase - 0.25, ybase - 0.25])

            final_xlines = []
            final_ylines = []
            for i in range(len(x_lines)):
                xi = x_lines[i]
                keep = True
                for j in range(len(x_lines)):
                    xj = x_lines[j]
                    if i != j and min(xi) >= min(xj) and max(xi) <= max(xj):
                        keep = False
                        break
                if keep:
                    final_xlines.append(x_lines[i])
                    final_ylines.append(y_lines[i])


            # final_ylines = []
            # ybase = i
            # for i in range(len(final_xlines)):
            #     xi = final_xlines[i]
            #     for j in range(i+1,len(final_xlines)):
            #         xj = final_xlines[j]
            #         # if i != j and ( 

            #         #     (min(xi) <= min(xj) and max(xi) >= min(xj)) or
            #         #     (min(xi) <= max(xj) and max(xi) >= max(xj))  
            #         #     ):
            #         #     ybase -= 0.1
            #     final_ylines.append([ybase - 0.25, ybase - 0.25])

            # for xi,yi in zip(x_lines, y_lines):

            #     keep = True
            #     for xj, yj in zip(x_lines, y_lines):
            #         if 

            for x,y in zip(final_xlines, final_ylines):
                plt.plot(x, y, color="black")
            #print(x_lines)
            #print(cliques)


        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        ax.invert_xaxis()
        if b == "None":
            ax.set_title("Rankings across all base ensembles")
        else:
            if b == "ExtraTreesClassifier":
                ax.set_title("Rankings for ExtraTrees as base ensemble".format(b))
            elif b == "RandomForestClassifier":
                ax.set_title("Rankings for RandomForest as base ensemble".format(b))
            else:
                ax.set_title("Rankings for HeterogenousForest as base ensemble".format(b))


        plt.show()

    #rank_df.plot(kind="scatter", x = "rank_mean", y = "kb")
    #tmp.fillna(1024)
    # print(tmp)
    # print(rank_df[["rank_mean"]])

    # for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    #     plt.plot(rank_df["rank_mean"], rank_df["kb"])

#print(df)
# %%
data = np.array([[ 8.82, 11.8 , 10.37, 12.08],
                     [ 8.92,  9.58, 10.59, 11.89],
                     [ 8.27, 11.46, 10.24, 11.6 ],
                     [ 8.83, 13.25,  8.33, 11.51]])

import scipy.stats as ss
tmp = ss.friedmanchisquare(*data.T).pvalue

print(tmp)


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

def memory(row):
    # Lets assume a native implementation with
    #  - unsigned int left, right
    #  - bool is_leaf
    #  - unsinged int feature_idx
    #  - float threshold
    #  - float pred[n_classes]
    #   ==> 4+4+1+4+4+4*n_classes = 17 + 4*n_classes
    if row["dataset"] == "letter":
        n_classes = 26
    elif row["dataset"] == "thyroid":
        n_classes = 3
    elif row["dataset"] == "covtype":
        n_classes = 7
    elif row["dataset"] == "wine-quality":
        n_classes = 6
    elif row["dataset"] == "pen-digits":
        n_classes = 10
    else:
        n_classes = 2

    return row["scores.mean_n_nodes"] * (17 + 4*n_classes)  / 1024.0

dataset = "multi"
#dataset = "adult"
#dataset = "bank"
#dataset = "connect"
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
df["KB"] = df.apply(memory, axis=1)

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
# if dataset == "letter":
#     n_classes = 26
# elif dataset == "thyroid":
#     n_classes = 3
# elif dataset == "covtype":
#     n_classes = 7
# elif dataset == "wine-quality":
#     n_classes = 6
# elif dataset == "pen-digits":
#     n_classes = 10
# else:
#     n_classes = 2
# df["KB"] = df["scores.mean_n_nodes"] * (17 + 4*n_classes)  / 1024.0

df = df.round(decimals = 3)
fdf = df.loc[df["n_estimators"] == 32]
fdf = fdf.loc[fdf["height"] == 5]
fdf = fdf.loc[fdf["dataset"] == "bank"]
#fdf = df.loc[df["KB"] < 512]

tabledf = fdf[["nice_name", "accuracy", "n_nodes", "fit_time", "n_estimators", "KB","height","dataset"]]
tabledf = tabledf.sort_values(by=['accuracy'], ascending = False)
print("Processed {} experiments".format(len(tabledf)))
display(HTML(tabledf.to_html()))

print("Best configuration per group")
#print(fdf.groupby(['model'])['accuracy'].count())
idx = fdf.groupby(['model'])['accuracy'].transform(max) == fdf['accuracy']
gdf = fdf[idx]

tabledf = gdf[["nice_name", "accuracy", "n_nodes", "fit_time", "n_estimators", "KB","height","dataset"]]
tabledf = tabledf.sort_values(by=['accuracy'], ascending = False)
print("Best configuration per group")
display(HTML(tabledf.to_html()))

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

# Sanitize names for later plotting
def nice_name(row):
    if row["model"] == "ExtraTreesClassifier":
        return "ET"
    elif row["model"] == "RandomForestClassifier":
        return "RF"
    elif row["model"] == "ProxPruningClassifier":
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
    else:
        return row["model"]

def highlight(s):
    accs = []
    kbs = []
    for i in range(0, len(s), 2):
        accs.append(s[i])
        kbs.append(s[i+1])

    max_acc = np.nanmax(accs)
    min_kb = np.nanmin(kbs)

    style = []
    for acc, kb in zip(accs, kbs):
        if acc == max_acc:
            style.append('background-color: blue; text-align: left')
        else:
            style.append('')
        
        if kb == min_kb:
            style.append('background-color: green')
        else:
            style.append('')
    return style

dataset = os.path.join(dataset, "results")
all_subdirs = [os.path.join(dataset,d) for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
#print(all_subdirs)
latest_folder = max(all_subdirs, key=os.path.getmtime)
#latest_folder = "eeg/results/19-02-2021-13:29:39"

print("Reading {}".format(os.path.join(latest_folder, "results.jsonl")))
df = read_jsonl(os.path.join(latest_folder, "results.jsonl"))
df["nice_name"] = df.apply(nice_name, axis=1)
df["KB"] = df.apply(memory, axis=1)

df["accuracy"] = df["scores.mean_accuracy"]
df["n_nodes"] = df["scores.mean_n_nodes"]
df["fit_time"] = df["scores.mean_fit_time"]
df["n_estimators"] = df["scores.mean_n_estimators"]

df["model"] = df.apply(nice_name, axis=1)
print(df["model"])
exit(1)
wdwdd
for kb in [16,32,64,128,256,512,None]:
    for b in ["RandomForestClassifier", "ExtraTreesClassifier", "HeterogenousForest", None]:
        dff = df.copy()
        
        # Filter for a specific base learner, but keep RF / ET / HF as well
        if b is not None:
            dff = dff.loc[
                ((dff["model"] == "RF") & (dff["base"] == "RandomForestClassifier")) |
                ((dff["model"] == "ET") & (dff["base"] == "ExtraTreesClassifier")) |
                ((dff["model"] == "HF") & (dff["base"] == "HeterogenousForest")) |
                (dff["base"] == b)
            ]
        
        # Filter for KB constraints
        if kb is not None:
            dff["accuracy"].loc[dff["KB"] > kb] = 0
        print(dff)
        dff = dff[["model", "accuracy", "KB", "dataset"]]
        print(dff)
        dff = dff.loc[  dff.groupby(["dataset", "model"])["accuracy"].idxmax() ]
        #print(dff)

        #print("Storing rankings for base {} and constrainted to {} KB".format(b, kb))
        #dff.reset_index().to_csv("results_{}_{}.csv".format(b, kb), index=False)

        #dff = dff.pivot_table(["KB", "accuracy"] , ['dataset'], 'model')
        #dff = dff.reorder_levels([1,0], axis=1).sort_index(axis=1).reindex(["accuracy", "KB"], level=1, axis=1)
        #display( dff )
        #display( dff.style.apply(highlight,axis=1) )
#print(df)


# %%

fdf = df
fdf = df.loc[df["base"] == "RandomForestClassifier"]
#fdf = df.loc[df["KB"] < 512]
#fdf = fdf.loc[df["model"] != "ExtraTreesClassifier"]
#fdf = fdf.loc[df["model"] != "ExtraTreesClassifier"]
#fdf = fdf.loc[df["height"] == 16]


tmp = fdf.groupby(['model']).apply(lambda x: x.sort_values(['accuracy'], ascending=False))
tmp = tmp.drop_duplicates(["dataset", "model"])
tmp = tmp[["model", "accuracy", "dataset"]]
#display(HTML(tmp.to_html()))
tmp.to_csv("rankings.csv",index=False)

tmp = fdf[["model", "accuracy", "KB", "dataset"]]
# tmp = tmp.groupby(['dataset']).apply(lambda x: x.sort_values(['accuracy'], ascending=False))
# tmp = tmp.drop_duplicates(["dataset", "model"])
tmp = tmp.groupby(["dataset", "model"]).max()
#print(tmp)
#tmp = tmp[["model", "accuracy", "KB"]].reset_index()
#tmp = tmp.drop(columns = ["level_1"])

tmp = df.pivot_table('accuracy', ['dataset'], 'model')
#tmp.reset_index( drop=False, inplace=True )
#tmp = tmp[["model", "accuracy", "KB", "dataset"]]
#tmp = tmp[["model", "accuracy", "KB","dataset"]]

display(HTML(tmp.to_html()))

# ranks = {}
# datasets = fdf["dataset"].unique()
# for d in datasets:
#     tdf = fdf.loc[fdf["dataset"] == d]
    
#     #print(maxkb)
#     #tdf = tdf.loc[tdf["KB"] < 0.005 * max(tdf["KB"].values)]
#     gdf = tdf.sort_values('accuracy', ascending=False).drop_duplicates(['model'])
    
#     #idx = tdf.groupby(['model'])['accuracy'].transform(max) == tdf['accuracy']
#     #gdf = tdf[idx].sort_values(by=['accuracy'], ascending = False)
    
#     # tabledf = gdf[["nice_name", "accuracy", "n_nodes", "fit_time", "n_estimators", "KB","height","dataset"]]
#     # tabledf = tabledf.sort_values(by=['accuracy'], ascending = False)
#     # print("Best configuration per group")
#     # display(HTML(tabledf.to_html()))


#     for i, (_, row) in enumerate(gdf.iterrows()):
#         if row["model"] in ranks:
#             ranks[row["model"]].append(i+1)
#         else:
#             ranks[row["model"]] = [i+1]

# print(datasets)
# print(ranks)
# print("RANKS ARE")
# for m, r in ranks.items():
#     print("{}: {} +- {}".format(m, np.mean(r), np.std(r)))

# gdf = tdf.sort_values('accuracy', ascending=False).drop_duplicates(['model'])

# %%
print("IMPORT")
import CriticalDifferenceDiagrams_jl as cdd # will take quite some time!
import pandas as pd
from wget import download
print("IMPORT DONE")

# we generate the above example from the underlying data
download("https://raw.githubusercontent.com/hfawaz/cd-diagram/master/example.csv")
df = pd.read_csv("example.csv")

plot = cdd.plot(
    df,
    "classifier_name", # the name of the treatment column
    "dataset_name",    # the name of the observation column
    "accuracy",        # the name of the outcome column
    maximize_outcome=True, # compute ranks for minimization (default) or maximization
    title="CriticalDifferenceDiagrams.jl" # give an optional title
)

# configure the preamble of PGFPlots.jl (optional)
cdd.pushPGFPlotsPreamble("\\usepackage{lmodern}")

# export to .svg, .tex, or .pdf
cdd.save("example.pdf", plot)

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
