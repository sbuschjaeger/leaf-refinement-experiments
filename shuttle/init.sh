#/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst

uncompress shuttle.trn.Z
cat shuttle.trn shuttle.tst > data.csv
