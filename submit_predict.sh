#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=1-10:15:00     # 1 day and 10 hours 15 minutes
#SBATCH --job-name="prediction"
####SBATCH -p wmalab
#SBATCH -p gpu # This is the default partition, you can use any of the following; intel, batch, highmem, gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --output=slurm-predict-%J.out

# Print current date
date

# Load samtools
#source activate tf_base
CHROMOSOME=${1} #19, 20, 21, 22, X
WIN_LEN=${2} # 200, 400, 80
source activate env_EnHiC

python test_predict.py ${CHROMOSOME} ${WIN_LEN}
