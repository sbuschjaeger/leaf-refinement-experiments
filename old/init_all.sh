#/bin/bash
dir=$(pwd)
#&& mkdir RandomForestClassifier && mv * RandomForestClassifier
for dataset in adult connect chess anura bank eeg elec postures japanese-vowels magic mozilla mnist nomao avila ida2016 satimage; do
    echo "Loading $dataset" 
    cd $dataset/results 
    ./init.sh
    cd $dir
done