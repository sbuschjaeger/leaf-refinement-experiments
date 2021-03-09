#/bin/bash

for dataset in adult bank connect covtype dry-beans eeg elec gas-drift japanese-vowels letter magic mozilla mushroom pen-digits satimage shuttle spambase thyroid wine-quality ; do
    print "Loading " $dataset
    cd $dataset
    ./init.sh
    cd ..
done