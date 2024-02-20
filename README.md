# Introduction

The code is author's implementation for paper "Pessimistic Causal Reinforcement Learning with Mediators for Confounded Observational Data".

# Requirements
First, create and activate `"pescal"` environment, and install required packages. Download the projec and run the following code in terminal under the root directory that contains `requirements.txt`.

```
conda create -n pescal python=3.7
conda activate pescal
pip install -r requirements.txt
```

# Reproduce figures in paper

The code is organized as follows: `"simulation"` folder contains the code to reproduce the simulation results in Figure 5; `"real"` folder contains the code to reproduce the real data experiment results in Figure 6. Results can be directly visualized by using the provided pre-trained data (in `daat` subfolder) in both `simulation` and `real` folder (please jump to "[Visualize Results](#visualize-results)" section below).

## Training
In `simulation` folder, run:
```
python simulation.py
```
In `real` folder, run:
```
python real.py
```
The code will generate and save training results of 600 `.json` files in a subfolder called `"data"`.

## Visualize Results

In `simulation` folder, run
```
python read_sim.py
```
In `real` folder, run 
```
python read_real.py
```
The code will generate all the plots in Figure 5 and Figure 6.

We use a computing cluster that has Sky Lake CPUs @ 2.60GHz, 24 processor cores, 96 GB of RAM, and 100 Gbps Infiniband interconnects. Runtime for PESCAL and CAL both take around 5 minutes for simulation; and around 1.5 hours for real data experiment, for 10000 steps of training of each seed.
