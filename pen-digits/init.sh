#/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra
wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes

cat pendigits.tra pendigits.tes > data.txt