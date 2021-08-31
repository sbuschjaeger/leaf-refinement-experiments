#/bin/bash

for dataset in adult anura bank chess connect eeg elec postures japanese-vowels magic mozilla mnist nomao avila ida2016 satimage; do
    print "Loading " $dataset
    cd $dataset
    ./init.sh
    cd ..
done