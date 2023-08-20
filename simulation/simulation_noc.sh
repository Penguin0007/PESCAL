#!/bin/bash 
#SBATCH -t 04:00:00
#SBATCH --nodes=1 
# Loads application and set up directory
module --force purge
module load anaconda
source activate pescal
cd /home/Downloads/simulation/

echo "$i$j"
python simulation_noc.py --iters ${1} --ratio ${2}

#terminal command: for i in `seq 1 100`; do for j in 0.0001 0.001 0.1 1; do sbatch simulation_noc.sh $i $j; done; done


