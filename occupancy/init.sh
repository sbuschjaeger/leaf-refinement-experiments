#/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip
unzip occupancy_data.zip

tail -n +2 datatest2.txt > tmp1.csv
tail -n +2 datatest.txt > tmp2.csv
cat datatraining.txt tmp1.csv tmp2.csv > data.csv

rm tmp1.csv
rm tmp2.csv