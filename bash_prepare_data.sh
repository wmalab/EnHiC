#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=1-10:15:00     # 1 day and 10 hours 15 minutes
#SBATCH --job-name="data"
#SBATCH --output=slurm-data-%J.out
#SBATCH -p wmalab

# Print current date
date

# Load samtools
#source activate tf_base
source activate env_EnHiC
# run
echo python test_preprocessing.py ${1} ${2} ${3}
python test_preprocessing.py ${1} ${2} ${3}

# Print name of node
hostname