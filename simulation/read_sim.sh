#!/bin/bash 
#SBATCH -t 04:00:00
#SBATCH --nodes=1 
# Loads application and set up directory
module --force purge
module load anaconda
source activate pescal
cd /home/Downloads/simulation/

echo "$i$j"
python read_sim.py --ratio ${1} --c ${2}

#terminal command: for i in 0.0001 0.001 0.1 1; do sbatch read_sim.sh $i noc; done





