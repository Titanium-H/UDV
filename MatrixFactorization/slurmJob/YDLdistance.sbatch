#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8000
#SBATCH --time 10:00:00

module load MATLAB/2021a

echo STARTING AT `date`
echo "running on: "
hostname

matlab -nodisplay -nodesktop -r "MC=${1}; noisy=${2}; xi=${3}; TestMain_distance; exit"

echo FINISHED at `date`