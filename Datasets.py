import tempfile
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.io.arff import loadarff
import urllib
import urllib.request

from io import BytesIO, TextIOWrapper
from zipfile import ZipFile

from tqdm import tqdm

def download(url, filename, tmpdir = None):
    """Download the file under the given url and store it in the given tmpdir udner the given filename. If tmpdir is None, then `tempfile.gettmpdir()` will be used which is most likely /tmp on Linux systems.

    Args:
        url (str): The URL to the file which should be downloaded.
        filename (str): The name under which the downlaoded while should be stored.
        tmpdir (Str, optional): The directory in which the file should be stored. Defaults to None.

    Returns:
        str: Returns the full path under which the file is stored. 
    """
    if tmpdir is None:
        tmpdir = os.path.join(tempfile.gettempdir(), "data")

    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    if not os.path.exists(os.path.join(tmpdir,filename)):
        print("{} not found. Downloading.".format(os.path.join(tmpdir,filename)))
        urllib.request.urlretrieve(url, os.path.join(tmpdir,filename))
    return os.path.join(tmpdir,filename)

def read_arff(path, class_name):
    """Loads the ARFF file under the given path and transforms it into a pandas dataframe. Each column which does not match class_name is copied into the pandas frame without changes. The column with the name `class_name` is renamed to `label` in the DataFrame. The behaviour of this method is undefined if the ARFF file already contains a `label` column and `class_name != 'label'`. 

    Args:
        path (str): The path to the ARFF file.
        class_name (str): The label column in the ARFF file

    Returns:
        pandas.DataFrame : A pandas dataframe containing the data from the ARFF file and an additional `label` column.
    """
    data, meta = loadarff(path)
    Xdict = {}
    for cname, ctype in zip(meta.names(), meta.types()):
        # Get the label attribute for the specific dataset:
        #   eeg: eyeDetection
        #   elec: class
        #   nomao: Class
        #   polish-bankruptcy: class
        if cname == class_name:
        #if cname in ["eyeDetection", "class",  "Class"]:
            enc = LabelEncoder()
            Xdict["label"] = enc.fit_transform(data[cname])
        else:
            Xdict[cname] = data[cname]
    return pd.DataFrame(Xdict)

def get_dataset(dataset, tmpdir = None):
    """Returns X,y of the given dataset by name. If the dataset does not exist it will be automatically downloaded.

    Args:
        dataset (str): The name of the dataset to be returned (and downloaded if required.). Currently supports {magic, mnist, fashion, adult, eeg}
        tmpdir ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: Raises a ValueError if an unsupported dataset is passed as an argument

    Returns:
        X, y (numpy array, numpay array): Returns the (N, d) dataset and the (N, ) labels where N is the number of data points and d is the number of features. 
    """

    if dataset == "magic":
        magic_path = download("http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data", "magic.csv", tmpdir)
        df = pd.read_csv(magic_path)
        X = df.values[:,:-1].astype(np.float64)
        Y = df.values[:,-1]
        Y = np.array([0 if y == 'g' else 1 for y in Y])
    elif dataset == "adult":
        adult_path = download("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.csv", tmpdir)

        col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
            "hours-per-week", "native-country", "label"
        ]
        df = pd.read_csv(adult_path, header=None, names=col_names)
        df = df.dropna()
        label = df.pop("label")
        Y = np.array([0 if l == " <=50K" else 1 for l in label])
        df = pd.get_dummies(df)
        X = df.values
    elif dataset == "shuttle":
        # wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z
        # wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst

        # uncompress shuttle.trn.Z
        # cat shuttle.trn shuttle.tst > data.csv
        shuttle_train = download("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z", "shuttle_train.Z", tmpdir)

        import unlzw3
        from pathlib import Path
        from io import StringIO
        s = unlzw3.unlzw(Path(shuttle_train)).decode("utf-8") 
        df_train = pd.read_csv(StringIO(s), delimiter=" ")
        Y_train = df_train.values[:,-1]
        Y_train = Y_train - min(Y_train)
        Y_train = np.array( [1 if y > 0 else 0 for y in Y_train] )
        X_train = df_train.values[:,:-1]

        shuttle_test = download("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst", "shuttle_test.csv", tmpdir)
        df_test = pd.read_csv(shuttle_test, delimiter=" ")
        Y_test = df_test.values[:,-1]
        Y_test = Y_test - min(Y_test)
        Y_test = np.array( [1 if y > 0 else 0 for y in Y_test] )
        X_test = df_test.values[:,:-1]

        return X_train,Y_train,X_test,Y_test
    elif dataset == "avila":
        avila_path = download("https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip", "avila.zip", tmpdir)

        zfile = ZipFile(avila_path, 'r')

        # cat avila/avila-ts.txt avila/avila-tr.txt > data.csv
        avila_test = TextIOWrapper(zfile.open("avila/avila-ts.txt"), encoding='ascii')
        df_test = pd.read_csv(avila_test, header=None)
        df_test = df_test.dropna()

        avila_train = TextIOWrapper(zfile.open("avila/avila-tr.txt"), encoding='ascii')
        df_train = pd.read_csv(avila_train, header=None)
        df_train = df_train.dropna()

        X_train = df_train.values[:,:-1].astype(np.float64)
        label = df_train.values[:,-1]
        le = LabelEncoder()
        Y_train = le.fit_transform(label)

        X_test = df_test.values[:,:-1].astype(np.float64)
        label = df_test.values[:,-1]
        Y_test = le.transform(label)

        return X_train,Y_train,X_test,Y_test
    elif dataset == "letter":
        letter_path = download("http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data", "letter.data", tmpdir)

        df = pd.read_csv(letter_path, header=None)
        X = df.values[:,1:].astype(np.float64)
        Y = df.values[:,0]
        Y = np.array( [ord(y) - 65 for y in Y] )
    elif dataset == "connect":
        connect_path = download("https://www.openml.org/data/get_csv/4965243/connect-4.arff", "connect.csv", tmpdir)
        
        df = pd.read_csv(connect_path, header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("class")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    elif dataset == "weather":
        weather_path = download("https://github.com/scikit-multiflow/streaming-datasets/blob/master/weather.csv?raw=true", "weather.csv", tmpdir)

        df = pd.read_csv(weather_path)
        Y = df["target"].values.astype(np.int32)
        df = df.drop(["target"], axis=1)
        X = df.values.astype(np.float64)
    elif dataset == "covtype":
        if tmpdir is None:
            out_path = os.path.join(tempfile.gettempdir(), "data", "covertype.csv.gz")
        else:
            out_path = os.path.join(tmpdir, "data", "covertype")
        data_path = download("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz", out_path)

        df = pd.read_csv(data_path)
        X = df.values[:,:-1].astype(np.float64)
        Y = df.values[:,-1] - 1 # Make sure classes start with index 0 instead of index 1 
    elif dataset == "bincovtype":
        if tmpdir is None:
            out_path = os.path.join(tempfile.gettempdir(), "data", "covertype.csv.gz")
        else:
            out_path = os.path.join(tmpdir, "data", "covertype")
        data_path = download("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz", out_path)

        df = pd.read_csv(data_path)
        Y = df.values[:,-1] - 1
        X = df.values[:,:-1].astype(np.float64)
        idx = np. where( (Y == 0) | (Y==1) )

        X = X[ idx ]
        Y = Y[ idx ]
    elif dataset == "fashion" or dataset == "mnist":
        def load_mnist(path, kind='train'):
            # Taken from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
            import os
            import gzip
            import numpy as np

            """Load MNIST data from `path`"""
            labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
            images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

            with gzip.open(labels_path, 'rb') as lbpath:
                labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

            with gzip.open(images_path, 'rb') as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

            return images, labels

        if dataset == "fashion":
            if tmpdir is None:
                out_path = os.path.join(tempfile.gettempdir(), "data", "fashion")
            else:
                out_path = os.path.join(tmpdir, "data", "fashion")

            train_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", out_path)
            train_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz", out_path)
            test_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", out_path)
            test_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz", out_path)
        else:
            if tmpdir is None:
                out_path = os.path.join(tempfile.gettempdir(), "data", "mnist")
            else:
                out_path = os.path.join(tmpdir, "data", "mnist")

            train_path = download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", out_path)
            train_path = download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz", out_path)
            test_path = download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", out_path)
            test_path = download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz", out_path)

        X_train, y_train = load_mnist(out_path, kind='train')
        X_test, y_test = load_mnist(out_path, kind='t10k')
        return X_train,y_train,X_test,y_test
    elif dataset == "eeg":
        eeg_path = download("https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff", "eeg.arff", tmpdir)
        
        df = read_arff(eeg_path, "eyeDetection")
        df = pd.get_dummies(df)
        df.dropna(axis=1, inplace=True)
        Y = df["label"].values.astype(np.int32)
        df = df.drop("label", axis=1)

        X = df.values.astype(np.float64)
    elif dataset == "elec":
        elec_path = download("https://github.com/hmgomes/StreamingRandomPatches/blob/master/datasets/elecNormNew.arff.zip?raw=true", "elecNormNew.arff.zip", tmpdir)
        zfile = ZipFile(elec_path, 'r')
        wrappedfile = TextIOWrapper(zfile.open('elecNormNew.arff'), encoding='ascii')
        data, meta = loadarff(wrappedfile)

        Xdict = {}
        for cname, ctype in zip(meta.names(), meta.types()):
            if cname == "class":
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
    elif dataset == "mozilla":
        mozilla_path = download("https://www.openml.org/data/get_csv/53929/mozilla4.arff", "mozilla.csv", tmpdir)
        df = pd.read_csv(mozilla_path, header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("state")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    elif dataset == "gas-drift":
        gas_path = download("https://www.openml.org/data/get_csv/1588715/phpbL6t4U.csv", "gas-drift.csv", tmpdir)

        df = pd.read_csv(gas_path, header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("Class")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    elif dataset == "bank":
        bank_path = download("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip", "bank.zip", tmpdir)
        zfile = ZipFile(bank_path, 'r')
        wrappedfile = TextIOWrapper(zfile.open('bank-full.csv'), encoding='ascii')
        df = pd.read_csv(wrappedfile, header=0, delimiter=";")
        df = df.dropna()
        label = df.pop("y")
        Y = np.array([0 if l == "no" else 1 for l in label])
        df = pd.get_dummies(df)
        X = df.values
    elif dataset == "nomao":
        bank_path = download("https://github.com/hmgomes/StreamingRandomPatches/blob/master/datasets/nomao.arff.zip?raw=true", "nomao.arff.zip", tmpdir)
        zfile = ZipFile(bank_path, 'r')
        wrappedfile = TextIOWrapper(zfile.open('nomao.arff.txt'), encoding='ascii')
        data, meta = loadarff(wrappedfile)
        
        Xdict = {}
        for cname, ctype in zip(meta.names(), meta.types()):
            if cname == "Class":
                enc = LabelEncoder()
                Xdict["label"] = enc.fit_transform(data[cname])
            else:
                Xdict[cname] = data[cname]
        df = pd.DataFrame(Xdict)
        df = pd.get_dummies(df)
        Y = df["label"].values.astype(np.int32)
        df = df.drop("label", axis=1)

        X = df.values.astype(np.float64)
    elif dataset == "japanese-vowels":
        jv_path = download("https://www.openml.org/data/get_csv/52415/JapaneseVowels.arff", "japanese-vowels.csv", tmpdir)
        df = pd.read_csv(jv_path, header = 0, delimiter=",")
        df = df.dropna()
        label = df.pop("speaker")
        le = LabelEncoder()
        Y = le.fit_transform(label)
        X = df.values
    else:
        raise ValueError("Unsupported dataset provided to get_dataset in datasets.py: {}!".format(dataset))
        # return None, None

    return X, Y
        