# Introduction

The code is author's implementation for "Pessimistic Causal Reinforcement Learning with Mediators for Confounded Observational Data".

# Requirements
First, create and activate `"pescal"` environment, and install required packages. Download and run the following code in terminal under the folder that contains `requirements.txt`.

```
module load anaconda
conda create -n pescal python=3.7
source activate pescal
pip install -r requirements.txt
```

# Figures in paper

The code is organized as follows: `"simulation"` folder contains the code to reproduce the simulation results in Figure 5; `"real"` folder contains the code to reproduce the real data experiment results in Figure 6. You can directly visualize results using the provided pre-trained data in both `simulation` and `real` folder. To avoid running training, and directly visualize results, please jump to "[Visualize results](#visualize-results)" session below.

## Training
In `simulation` folder, run:
```
python simulation.py
```
In `real` folder, run:
```
python real.py
```
The code will generate and save training results of 600 `.json` files in a folder called `"data"`.

## Visualize results

Under `simulation` folder, run
```
python read_sim.py
```
Under `real` folder, run 
```
python read_real.py
```
The code will generate all the plots in Figure 5 and Figure 6.

We use a computing cluster that has Sky Lake CPUs @ 2.60GHz, 24 processor cores, 96 GB of RAM, and 100 Gbps Infiniband interconnects. Runtime for PESCAL and CAL both takes around 5 minutes for simulation; and around 1.5 hours for real data experiment, for running 10000 training steps in each seed.
