#!/bin/bash

for MC in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    for NOISY in 0 1; do
        for STEPSIZE in 0.1 0.01 0.001; do
            echo Submitting job ${DATA}
            sbatch YDLstepsize.sbatch ${MC} ${NOISY} ${STEPSIZE}
            echo Submitted job
            sleep 0.1; # pause a bit to get a new random seed
        done
    done
done