#/bin/bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00414/to_uci.zip

unzip to_uci.zip

tail -n +22 to_uci/aps_failure_test_set.csv > tmp1.csv
cat to_uci/aps_failure_training_set.csv tmp1.csv > data.csv
rm tmp1.csv