#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=90G
#SBATCH --time=20-10:15:00     # 20 day and 10 hours 15 minutes
#SBATCH --job-name="gan_training"
####SBATCH -p wmalab
#SBATCH -p gpu # This is the default partition, you can use any of the following; intel, batch, highmem, gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --output=slurm-train-%J.out

# Print current date
date

# Print name of node
hostname

# Load samtools
source activate env_EnHiC

# run
#the size of input
len_size=${1}
max_distance=${2}
python train.py $len_size $max_distance

