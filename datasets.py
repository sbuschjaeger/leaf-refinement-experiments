import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.io.arff import loadarff

def get_dataset(dataset):
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
        # df = pd.read_csv(os.path.join(dataset,"mnist_test.csv"), header = 0, delimiter=",")
        # df = df.dropna()
        # XTest = df.values[:,1:]
        # YTest = le.transform(df.values[:,0])
        # X = (XTrain, XTest)
        # Y = (YTrain, YTest)
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
    elif dataset == "weather":
        df = pd.read_csv(os.path.join(dataset,"weather.csv"))
        Y = df["target"].values.astype(np.int32)
        df = df.drop(["target"], axis=1)
        X = df.values.astype(np.float64)
    elif dataset == "dota2":
        df = pd.read_csv(os.path.join(dataset, "data.csv"), header=None)
        df = df.dropna()
        label = df.values[:,0]
        X = df.values[:,1:]
        le = LabelEncoder()
        Y = le.fit_transform(label)
    elif dataset == "chess":
        df = pd.read_csv(os.path.join(dataset, "krkopt.data"), header=None)
        df = df.dropna()
        label = df[df.columns[-1]]
        df = df[df.columns[:-1]]
        df = pd.get_dummies(df)
        X = df.values
        le = LabelEncoder()
        Y = le.fit_transform(label)
    else:
        return None, None

    return X, Y
        