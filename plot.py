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



def nice_name(row):
    '''
    Sanitize names for later plotting. 
    '''
    if row["pruning"] == "ExtraTreesClassifier":
        return "ET"
    elif row["pruning"] == "AdaBoostClassifier":
        return "AB"
    elif row["pruning"] == "GradientBoostingClassifier":
        return "GB"
    elif row["pruning"] == "RandomForestClassifier":
        return "RF"
    elif row["pruning"] == "BaggingClassifier":
        return "Bag."
    # elif row["model"] == "LeafRefinement" or row["model"] == "ProxPruningClassifier": # We changed the name after we already did 75% of the experiments. 
    #     if base == "RandomForestClassifier":
    #         return "RF-LR"
    #     elif base == "BaggingClassifier":
    #         return "Bag-LR"
    #     else:
    #         return "ET-LR"
    elif row["pruning"] == "RandomPruningClassifier":
        return "rand."
    elif row["pruning"] == "complementariness":
        return "COMP"
    elif row["pruning"] == "drep":
        return "DREP"
    elif row["pruning"] == "individual_contribution":
        return "IC"
    elif row["pruning"] == "individual_error":
        return "IE"
    elif row["pruning"] == "individual_kappa_statistic":
        return "IKS"
    elif row["pruning"] == "individual_margin_diversity":
        return "IMD"
    elif row["pruning"] == "margin_distance":
        return "MD"
    elif row["pruning"] == "reduced_error":
        return "RE"
    elif row["pruning"] == "HeterogenousForest":
        return "HF"
    elif row["pruning"] == "reference_vector":
        return "RV"
    elif row["pruning"] == "error_ambiguity":
        return "EA"
    elif row["pruning"] == "largest_mean_distance":
        return "LMD"
    elif row["pruning"] == "cluster_accuracy":
        return "CA"
    elif row["pruning"] == "cluster_centroids":
        return "CC"
    elif row["pruning"] == "combined_error":
        return "CE"
    elif row["pruning"] == "combined":
        return "comb."
    else:
        return row["pruning"]

def method_name(row):
    if row["pruning"] == "random" and row["LR"]:
        return "LR"
    elif row["pruning"] == "random" and not row["LR"]:
        return "RF"
    else:
        return row["pruning"] + ("+LR" if row["LR"] is True else "")

files = [
    "adult_22-03-2022-19:00.csv", 
    "anuran_24-03-2022-10:20.csv", 
    "avila_23-03-2022-09:08.csv",
    "bank_22-03-2022-21:39.csv", 
    "connect_24-03-2022-04:51.csv",
    "eeg_22-03-2022-13:13.csv", 
    "elec_22-03-2022-15:51.csv", 
    "fashion_23-03-2022-16:46.csv",
    "gas-drift_23-03-2022-00:33.csv", 
    "ida2016_25-03-2022-09:41.csv",
    "japanese-vowels_23-03-2022-22:57.csv", 
    "magic_22-03-2022-16:58.csv", 
    "mnist_23-03-2022-21:00.csv",
    "mozilla_23-03-2022-02:49.csv", 
    "postures_25-03-2022-02:23.csv",
    # "29-03-2022-23:47.csv"
    #"shuttle_24-03-2022-16:55.csv", # Does this make sense?
    #"weather_23-03-2022-01:46.csv",
    #"28-03-2022-02:09.csv" # No L0 / hard-L0
]

dfs = []
for f in files:
    dataset = f.split("_")[0]
    df = pd.read_csv(f)
    df["pruning"] = df.apply(partial(nice_name), axis=1)
    df["method"] = df.apply(method_name, axis=1)
    dfs.append(df)
df = pd.concat(dfs, axis=0)
print("Read {} experiments".format(len(df)))
# df = df.loc[df["dataset"] != "weather"]
# df = df.loc[df["dataset"] != "shuttle"]
df

#%%

# def highlight_max(row):
#     max_val = row.max()
#     return ['font-weight: bold' if r == max_val else '' for r in row]

def add_memory(row):
    return str(np.round(row["size_kb"]/1024.0,2)) + " MB"

def add_metric(row,metric="accuracy"):
    if metric == "accuracy":
        return str(np.round(row[metric],2)) + " \%"
    else:
        return str(np.round(row[metric],4)) 
    # if metric == "accuracy":
    #     return str(np.round(row[metric],2)) + " (" + str( np.round(row["size_kb"]/1024.0,2)) + " MB)"
    # else:
    #     return str(np.round(row[metric],4)) + " (" + str( np.round(row["size_kb"]/1024.0,2)) + " MB)"

    #print(row)
    #dwd

for metric in ["accuracy", "f1"]:
    dff = df.copy()
    #dff = dff.loc[dff["method"] != "RF"]
    dff.sort_values([metric], inplace=True)
    dff = dff.drop_duplicates(["method", "dataset"], keep="last")
    dff.to_csv("cd_{}.csv".format(metric))
    tmp = dff.sort_values(["dataset"])[["dataset", "method","size_kb", metric]]
    tmp["memory"] = tmp.apply(add_memory, axis=1)
    tmp[metric] = tmp.apply(add_metric, metric=metric, axis=1)
    #tmp["combined"] = tmp.apply(add_memory, metric = metric, axis=1)
    #display(tmp)
    #ptable = tmp.pivot_table(index=["dataset"], values=["combined"], columns=["method"],aggfunc=lambda x: ' '.join(x))#.round(4)

    ptable = tmp.pivot_table(index=["dataset"], values=[metric,"memory"], columns=["method"], aggfunc=lambda x: x).round(4)
    ptable = ptable.swaplevel(0, 1, axis=1).sort_index(axis=1).stack()
    #dwdwd

    if metric == "f1":
        s = ptable.style.format(na_rep="-", precision=4).highlight_max(axis=1,props='bfseries: ;').hide_index(level=1)
    else:
        s = ptable.style.format(na_rep="-", precision=2).highlight_max(axis=1,props='bfseries: ;').hide_index(level=1)
    tex = s.to_latex()#float_format="%.2f", index=True)
    #tex = ptable.style.apply(highlight_max,axis=1).render().to_latex(na_rep="-", float_format="%.2f", index=True)

    #tex = ptable.to_latex(na_rep="-", float_format="%.2f", index=True)
    print("Best ".format(metric))
    print(tex)
    display(ptable)

#%%
def get_pareto(df, columns):
    ''' Computes the pareto front of the given columns in the given dataframe. Returns results as a dataframe.
    '''
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

colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

markers = ["o", "v", "^", "<", ">", "s", "P", "X", "D", "o", "v", "^", "<"]
styles = ["-", "--", "-.", ":","-", "--", "-.", ":","-", "--", "-.", ":","-", "--", "-."] 
style_mapping = {}

print(df["method"].unique())
for m,color,mark,style in zip(df["method"].unique(), colors, markers, styles):
    style_mapping[m] = {
        "color":color,
        "marker":mark,
        "style":style,
    }

# Q2: What methods offers the best accuracy memory tradeoff => CD Diagram
for metric in ["accuracy"]: #, "f1"
    aucs = []
    for d in df["dataset"].unique():
        dff = df.loc[df["dataset"] == d].copy()

        dff = dff.loc[dff["method"] != "RF"]
        #dff = dff.loc[dff["method"] != "L1"]
        max_kb = dff["size_kb"].max()
        # max_kb = 2048
        # dff = dff.loc[dff["size_kb"] <= max_kb]

        fig = plt.figure()
        names = []
        for name, group in dff.groupby(["method"]):
            marker, color, style = style_mapping[name]["marker"], style_mapping[name]["color"], style_mapping[name]["style"]
            pdf = get_pareto(group, [metric, "size_kb"])
            pdf = pdf[["method", metric, "size_kb", "train_time_sec"]]
            # pdf = pdf.loc[pdf["KB"] >= min_kb]
            pdf = pdf.sort_values(by=[metric], ascending = True)
            x = np.append(pdf["size_kb"].values, [max_kb])
            y = np.append(pdf[metric].values, [pdf[metric].max()]) / 100.0

            plt.scatter(x,y,s = [2.5**2 for _ in x], color = color)
            plt.plot(x,y, label=name, color=color) #marker=marker
            names.append(name)
            aucs.append(
                {
                    "method":name,
                    #"AUC":np.trapz(y, x),
                    "AUC":np.trapz(y, x) / max_kb,
                    "dataset":d
                }
            )

        print("{}".format(d))
        plt.legend(names)
        plt.xlabel("Model size [KB]")
        plt.ylabel(metric)
        plt.show()

    tabledf = pd.DataFrame(aucs)
    tabledf.sort_values(by=["AUC"], inplace = True, ascending=False)
    tabledf.to_csv("cd_auc_{}.csv".format(metric),index=False)
    display(tabledf)

# %%

# Q2a: What method offers the best accuracy given a certain budget? => 2d CD Diagram
for metric in ["accuracy", "f1"]:
    dfs = []
    mem_sizes = [128, 256, 512, 1024, 2048]

    all_methods = dff.loc[dff["method"] != "RF"]["method"].unique()
    for mem in mem_sizes:
        tmp_df = []
        for d in df["dataset"].unique():
            dff = df.loc[df["dataset"] == d]
            #dff = dff.loc[dff["method"] != "L1"]
            dff = dff.loc[dff["method"] != "RF"]
            
            mdf = dff.loc[dff["size_kb"] <= mem].copy()
            mdf.sort_values([metric], inplace=True)
            mdf = mdf.drop_duplicates(["method"], keep="last")
            mdf = mdf[["dataset", "method", metric]]

            row = {"dataset":d, "method":m, metric:0}
            for m in all_methods:
                if m not in mdf["method"].unique():
                    mdf = mdf.append(row, ignore_index = True)
            
            tmp_df.append(mdf)
        filtered_df = pd.concat(tmp_df, axis=0)

        tmp = filtered_df.sort_values(["dataset"])[["dataset", "method", metric]]
        ptable = tmp.pivot_table(index=["dataset"], values=[metric], columns=["method"]).round(4)

        if metric == "f1":
            s = ptable.style.format(na_rep="-", precision=4).highlight_max(axis=1,props='bfseries: ;')
        else:
            s = ptable.style.format(na_rep="-", precision=2).highlight_max(axis=1,props='bfseries: ;')
        tex = s.to_latex()


        filtered_df.to_csv("mem_{}_{}_kb.csv".format(mem,metric),index=False)
        print("Filtered for {} KB. Now {} entries".format(mem, len(filtered_df)))
        print(tex)

        # print(filtered_df[["method", "accuracy", "dataset"]])
        # print("")

# %%

# Q4: Ablation study: How does the tree size affect each method? => 2d CD Diagram
for metric in ["accuracy", "f1"]:
    dfs = []
    max_leafs = dff["max_leafs"].max()
    leaf_sizes = dff["max_leafs"].unique()

    for l in leaf_sizes:
        tmp_df = []
        for d in df["dataset"].unique():
            #print("{}".format(d))
            dff = df.loc[df["dataset"] == d]
            dff = dff.loc[dff["method"] != "RF"]
            #dff = dff.loc[dff["method"] != "L1"]
            
            mdf = dff.loc[dff["max_leafs"] <= l].copy()
            mdf.sort_values([metric], inplace=True)
            mdf = mdf.drop_duplicates(["method"], keep="last")
            mdf = mdf[["dataset", "method", metric]]
            
            tmp_df.append(mdf)
        filtered_df = pd.concat(tmp_df, axis=0)
        filtered_df.to_csv("leaf_{}_{}.csv".format(l, metric),index=False)
        print("Filtered for {} leafs. Now {} entries".format(l, len(filtered_df)))
    #print("")

# %%

# 16,32,64,128,256,512,1024,2048
# dff = df.loc[df["max_leafs"] == 16].copy()
dff = df.loc[df["method"] == "L1+LR"].copy()
# dff = dff.loc[dff["max_leafs"] == 256]
# dff = dff.loc[dff["dataset"] == "elec"]
# dff.loc[dff["l1"] == 0.05]["n_estimators"]

lambdas = []
mean_n_estimators = []
mean_acc = []
std_n_estimators = []
std_acc = []
min_n_estimators = []
max_n_estimators = []

for l in sorted(dff["l1"].unique()):
    #print(dff.loc[dff["l1"] == l]["n_estimators"])
    tmp = dff.loc[dff["l1"] == l]["size_kb"]
    mean_n_estimators.append(np.mean(tmp))
    min_n_estimators.append(np.min(tmp))
    max_n_estimators.append(np.max(tmp))
    std_n_estimators.append(np.std(tmp))
    mean_acc.append(np.mean(dff.loc[dff["l1"] == l]["accuracy"]))
    std_acc.append(np.std(dff.loc[dff["l1"] == l]["accuracy"]))
    lambdas.append(dff.loc[dff["l1"] == l]["n_estimators"])

# print(lambdas)
# print(mean_n_estimators)
# print(std_n_estimators)
# print(min_n_estimators)
# print(max_n_estimators)

mean_n_estimators = np.array(mean_n_estimators)
std_n_estimators = np.array(std_n_estimators)
min_n_estimators = np.array(min_n_estimators)
max_n_estimators = np.array(max_n_estimators)
mean_acc = np.array(mean_acc)
std_acc = np.array(std_acc)

#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams['text.usetex'] = False
# import numpy as np

# Visualize the result
plt.plot(lambdas, mean_n_estimators, 'ob', alpha=0.6)
#plt.plot(lambdas, sorted(df["K"].unique()), '+g', alpha=0.6)
plt.plot(lambdas, mean_n_estimators, '-', color='b', alpha=0.6)
plt.fill_between(lambdas, mean_n_estimators - std_n_estimators, mean_n_estimators + std_n_estimators,color='blue', alpha=0.1)
#plt.xlabel(r'\lambda')
plt.xlabel(r'λ')
plt.ylabel("Size")
plt.savefig("nest_over_lambda.pdf", format="pdf", bbox_inches="tight")
plt.show()

# plt.plot(lambdas, mean_acc, 'or')
# plt.plot(lambdas, mean_acc, '-', color='gray')
# plt.fill_between(lambdas, mean_acc - std_acc, mean_acc + std_acc,color='gray', alpha=0.2)
# plt.show()
# #plt.xlim(0, 10);

# %%

datasets = "datasets_25-03-2022-11:12.csv"
ddf = pd.read_csv(datasets)
ddf = ddf.drop_duplicates(["dataset"], keep="last")

tmp = ddf.to_latex(na_rep="-", float_format="%.2f", index=False)
print(tmp)

# Q2: What method offers the best accuracy given a certain tree depth? => 2d CD Diagram
# for d in df["dataset"].unique():
#     print("{}".format(d))
#     dff = df.loc[df["dataset"] == d].copy()

#     dff = dff.loc[dff["method"] != "RF"]
#     max_leafs = dff["max_leafs"].max()
#     leaf_sizes = dff["max_leafs"].unique()

#     for l in leaf_sizes:
#         mdf = dff.loc[dff["max_leafs"] <= l].copy()
#         mdf.sort_values(["accuracy"], inplace=True)
#         mdf = mdf.drop_duplicates(["method"], keep="last")
#         print(mdf[["method", "train_time_sec", "size_kb","f1", "accuracy"]])
#         print("")

# # %%
# # df = df.loc[df["pruning"] == "random"]
# dff = df.copy()
# dff.sort_values(["accuracy"], inplace=True)
# dff = dff.drop_duplicates(["method"], keep="last")
# #df = df.sort_values(by=["accuracy"])
# #with pd.option_context('display.max_rows', None): 
# dff[["dataset", "method", "train_time_sec", "size_kb","n_nodes","f1", "accuracy"]]


# # %%

# fig = plt.figure()
# methods = ["random", "RE"]
# with_lr = False
# colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
# names = []
# for d,c in zip(df["max_depth"].unique(), colors):
#     for mt,s in zip(methods, ["solid", "dashed"]):
#         dff = df.copy()
#         dff = dff.loc[dff["max_depth"] == d]
#         dff = dff.loc[dff["pruning"] == mt]
#         dff = dff.loc[dff["LR"] == with_lr]

#         dff.sort_values(["n_estimators"], inplace=True)
    
#         plt.plot(dff["n_estimators"].values, dff["accuracy"].values, color=c, linestyle=s)
#         names.append(r'{} $d={}$'.format(mt,d))
    
# plt.legend(names, loc="upper right", bbox_to_anchor=(1.32, 1))
# plt.xlabel("Number of trees")
# plt.ylabel("Accuracy")
# # if show:
# plt.show()
# # fig.savefig(os.path.join("plots","{}{}_{}_revisited.pdf".format(base,"_with_prune" if with_prune else "",d)), bbox_inches='tight')
# plt.close()