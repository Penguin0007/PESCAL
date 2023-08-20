#!/bin/bash 
#SBATCH -A standby
#SBATCH -t 04:00:00
#SBATCH --nodes=1
# Loads application and set up directory
module --force purge
module load anaconda
source activate pescal
cd /home/Downloads/real/

echo "$i$j"
python real.py --iters ${1} --ratio ${2}

#terminal command: for i in `seq 1 100`; do for j in 0.0003 0.001 0.1 1; do sbatch real.sh $i $j; done; done