#/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip
unzip avila.zip

cat avila/avila-ts.txt avila/avila-tr.txt > data.csv