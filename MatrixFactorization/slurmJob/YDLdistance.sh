#!/bin/bash

for MC in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    for NOISY in 0 1; do
        for XI in 0.0001 0.001 0.01 0.1 1; do
            echo Submitting job ${DATA}
            sbatch YDLdistance.sbatch ${MC} ${NOISY} ${XI}
            echo Submitted job
            sleep 0.1; # pause a bit to get a new random seed
        done
    done
done