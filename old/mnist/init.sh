#/bin/bash

wget https://pjreddie.com/media/files/mnist_train.csv
wget https://pjreddie.com/media/files/mnist_test.csv

cat mnist_test.csv mnist_train.csv > data.csv
