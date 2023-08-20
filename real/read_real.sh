#!/bin/bash 
#SBATCH -t 04:00:00
#SBATCH --nodes=1 
# Loads application and set up directory
module --force purge
module load anaconda
source activate pescal
cd /home/Downloads/real/

echo "$i$j$k$l$m$n"
python read_real.py --ratio ${1} --types ${2} ${3} ${4} ${5} ${6}

#terminal command: for i in 0.0003 0.001 0.1 1; do sbatch read_real.sh $i noc; done


