#!/bin/bash

reducers=( "tsne" "umap" )

for i in {1..5}
do
    for reducer in "${reducers[@]}"
    do
        python3 src/analysis.py --data data --reducer $reducer --plotName plots/$reducer/"$reducer"PCA"$i" --variance 0.9
    done
done
