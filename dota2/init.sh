#/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip

unzip dota2Dataset.zip

cat dota2Train.csv dota2Test.csv > data.csv