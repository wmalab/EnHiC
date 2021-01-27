#!/bin/bash
#SBATCH --partition=wmalab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=1-10:00:00
#SBATCH --job-name=tf_board
#SBATCH --output=slurm-tfboard-%J.out

#rm tfboard-*.out
# Change to HOME dir to give access to all folders within Jupyter-Lab
cd $HOME

# Jupyter vars
XDG_RUNTIME_DIR=""

# Get tunneling info
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# Print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -NL ${port}:${node}:${port} ${user}@cluster.hpcc.ucr.edu

MS Windows MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: cluster.hpcc.ucr.edu
SSH login: $user
SSH port: 22
"

# load modules or conda environments here
source activate env_EnHiC
echo -e "PLEASE USE GENERATED URL BELOW IN BROWSER\nYOU MUST REPLACE '${node}' with 'localhost'"
tensorboard --logdir='/rhome/yhu/bigdata/proj/demo_test/refine_resolution/logs/model/' --port=${port} --host=${node} --samples_per_plugin images=50
# Comment out the line above and uncomment the line below if you would like jupyter-notebook instead of jupyter-lab
#jupyter-notebook --no-browser --port=${port} --ip=${node}

