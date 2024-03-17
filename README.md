# Introduction

The author's code implementation for reproducing the experimental results in paper "Pessimistic Causal Reinforcement Learning with Mediators for Confounded Offline Data".

# Requirements
First, create and activate `"pescal"` environment, and install required packages. Download the repository and run the following code in terminal under the root directory that contains `requirements.txt`.

```
conda create -n pescal python=3.7
conda activate pescal
pip install -r requirements.txt
```

# Reproduce figures in paper

The code is organized as follows: `"simulation"` folder contains the code to reproduce the simulation results in Figure 5; `"real"` folder contains the code to reproduce the real data experiment results in Figure 6. Results can be directly visualized by using the provided pre-trained data, in `data` subfolder under both `simulation` and `real` folders (please jump to "[Visualize Results](#visualize-results)" section below).

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

We use a computing cluster that has Sky Lake CPUs @ 2.60GHz, 24 processor cores, 96 GB of RAM, and 100 Gbps Infiniband interconnects. The runtime for PESCAL and CAL is approximately 5 minutes for simulations and about 1.5 hours for real-data experiments. These runtimes are calculated over 10,000 training steps.
