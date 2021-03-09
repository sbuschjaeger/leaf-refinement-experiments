#/bin/bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

tail -n +2 winequality-white.csv > tmp.csv
cat winequality-red.csv tmp.csv > data.csv
rm tmp.csv
