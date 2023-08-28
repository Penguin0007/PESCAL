# Introduction

The code is for our submission "Pessimistic Causal Reinforcement Learning with Mediators for Confounded Offline Data". Code was run based on a cluster with slurm workload manager, with Anaconda available.

# Requirements
First, create and activate `"pescal"` environment, and install required packages, by running the following code in terminal under the path that contains the  `requirements.txt`.

```
module load anaconda
conda create -n pescal python=3.7
source activate pescal
pip install -r requirements.txt
```

# Figures in paper

The code is orginazed as follows: `"simulation"` folder contains the code to reproduce the simulation results in Figure 5; `"real"` folder contains the code to reproduce the real data experiment results in Figure 6.

## Figure 5
### Training
Python files `simulation_noc.py` and `simulation.py` in `simulation` folder contains code to run and generate training results for confounded mediated Markov Decision Process (M2DP, line 1 in Figure 5), and non-confounded mediated Markov Decision Process (Standard MDP, line 2 in Figure 5). They both have the following parameters to specify:

```
--iters: Iteration number, we used 100.
--rwindow: Moving window length of which we take average of emperical online reward, default 50.
--ratio: Ratio of size of original data (size of 50000) we want to keep. We used 0.0003, 0.5, 1 (3 settings).
```
In terminal, run `cd simulation` to set the working environment to `simulation` folder. Change the following line
```
cd /home/Downloads/simulation/
```

in the 3 files: `simulation.sh`, `simulation_noc.sh`, and `read_sim.sh` under `simulation` folder to your corresponding `simulation` folder path. Then run
```
for i in `seq 1 100`; do for j in 0.0003 0.5 1; do sbatch simulation.sh $i $j; done; done
for i in `seq 1 100`; do for j in 0.0003 0.5 1; do sbatch simulation_noc.sh $i $j; done; done
```
to submit $2\times3\times100$ slurm jobs for the four settings for M2DP and standard MDP, respectively. The code will generate and save training results of 600 `.json` files for both M2DP and standard MDP in a folder called `"data"`.

### Visualize results

In terminal under `simulation` folder, run
```
for i in 0.0003 0.5 1; do sbatch read_sim.sh $i noc; done
```
The submitted job will generate all the plots in Figure 5 for M2DP and standard MDP, called `pescal_sim_XXX.pdf` and `pescal_sim_noc_XXX.pdf`, where `XXX` takes different values of 0.0003, 0.5, and 1.

## Figure 6
The `Training` and `Visualize results` sessions in Figure 6 follow the same flavor as in Figure 5.
### Training
Python files `real_noc.py` and `real.py` in `real` folder contains code to run and generate training results for confounded mediated Markov Decision Process (M2DP, line 1 in Figure 6), and non-confounded mediated Markov Decision Process (Standard MDP, line 2 in Figure 6). They both have the following parameters to specify:

```
--iters: Iteration number, we used 100.
--rwindow: Moving window length of which we take average of emperical online reward, default 50.
--ratio: Ratio of size of original data (size of 50000) we want to keep. We used 0.0003, 0.5, 1 (3 settings).
```
In terminal, run `cd real` to set the working environment to `real` folder. Change the following line
```
cd /home/Downloads/real/
```

in the 3 files: `real.sh`, `real_noc.sh`, and `read_real.sh` under `real` folder to your corresponding `real` folder path. Then run
```
for i in `seq 1 100`; do for j in 0.0003 0.5 1; do sbatch real.sh $i $j; done; done
for i in `seq 1 100`; do for j in 0.0003 0.5 1; do sbatch real_noc.sh $i $j; done; done
```
to submit $2\times3\times100$ slurm jobs for the four settings for M2DP and standard MDP, respectively. The code will generate and save training results of 600 `.json` files for both M2DP and standard MDP in a folder called `"data"`.

### Visualize results

In terminal of `real` folder, run
```
for i in 0.0003 0.5 1; do sbatch read_real.sh $i noc; done
```
The submitted job will generate all the plots in Figure 6 for M2DP and standard MDP, which are called `pescal_real_XXX.pdf` and `pescal_real_noc_XXX.pdf`, where `XXX` takes values 0.0003, 0.5, and 1.


We used a computing cluster that has Two Sky Lake CPUs @ 2.60GHz, 24 processor cores, 96 GB of RAM, and 100 Gbps Infiniband interconnects. For each of the 600 jobs submitted for training, simulation takes 10~15 minutes, real data experiment takes about 3 hours to 3.5 hours; It takes within 5 minutes in `Visualize results` for drawing all the plots for both simulation and real data experiment.
