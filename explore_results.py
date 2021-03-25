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
        # return "HEP"
        if row["model_params.update_leaves"]:
            return "HEP-LR λ = {}".format(row["model_params.l_tree_reg"])
        else:
            return "HEP λ = {}".format(row["model_params.l_tree_reg"])
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
        
        dff = dff.loc[
            dff["model"].str.contains("HEP") 
        ]
        
        # Filter for KB constraints
        if kb is not None:
            dff["accuracy"].loc[dff["KB"] > kb] = 0
            dff["KB"].loc[dff["KB"] > kb] = 0
        #dff = dff.fillna(value = 0)

        dff = dff[["model", "accuracy", "KB", "dataset"]]
        dff.fillna(0, inplace=True)
        dff = dff.loc[  dff.groupby(["dataset", "model"])["accuracy"].idxmax() ]
        dff["rank"] = dff.groupby("dataset")["accuracy"].rank(axis=0, ascending=False)

        tabledf = dff.pivot_table(["KB", "accuracy", "rank"] , ['dataset'], 'model')
        tabledf = tabledf.reorder_levels([1,0], axis=1).sort_index(axis=1).reindex(["accuracy", "KB", "rank"], level=1, axis=1)
        tabledf.fillna(0, inplace=True)
        tabledf.to_latex(buf="raw_{}_{}.tex".format(b,kb),index=False, float_format="%.3f")

        if plot:
            ranks = dff.pivot_table(["rank"] , ['dataset'], 'model')
            ranks = ranks.T.fillna(ranks.max(axis=1) + 1).T
            m_order = [r[1] for r in ranks.columns]
            ranks = ranks.to_numpy().T
            ftest = ss.friedmanchisquare(*ranks)

            if ftest.pvalue < pval:
                tmp = sp.posthoc_wilcoxon(ranks,p_adjust="holm", zero_method="zsplit") 
                adj = tmp.values
                adj[adj > pval] = 1.0
                adj[adj <= pval] = 0.0
                cur_cliques = []
                for i, mi in enumerate(m_order):
                    clique = []
                    for j, mj in enumerate(m_order):
                        if i == j:
                            clique.append(mi)
                        elif adj[i][j] == 1.0:
                            clique.append(mj)
                    
                    cur_cliques.append(clique)

                final_cliques = cur_cliques
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

        fig.savefig("ranks_{}.pdf".format(b), bbox_inches='tight')
        plt.show()

print("DONE")

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