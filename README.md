# Imitation Improvement Learning for Large-scale Capacitated Vehicle Routing Problems

This repository contains the source code of the paper "Imitation Improvement Learning for Large-scale Capacitated Vehicle Routing Problems".

## Abstract

Recent works using deep reinforcement learning (RL) to solve routing problems such as the capacitated vehicle routing problem (CVRP) have focused on improvement learning-based methods, which involve improving a given solution until it becomes  near-optimal. Although adequate solutions can be achieved for small problem instances, their efficiency degrades for large-scale ones. 
In this work, we propose a new improvement learning-based framework based on imitation learning where classical heuristics serve as experts to  encourage the policy model to mimic and produce similar or better solutions. Moreover, to improve scalability,  we propose Clockwise Clustering, a novel augmented framework for decomposing large-scale CVRP into subproblems by clustering sequentially nodes in clockwise order, and then learning to solve them simultaneously.
Our approaches enhance state-of-the-art CVRP solvers while attaining competitive solution quality on several well-known datasets,  including real-world instances with sizes up to 30,000 nodes.  
Our best methods are able to achieve new state-of-the-art results for several large instances and generalize to a wide range of CVRP variants and solvers.
We also contribute new datasets and results to test the generalizability of our deep RL algorithms.

## Requirements
 - torch>=1.12.1
 - torch_geometric>=2.1.0
 - apex (optional)
 - VRP heuristic c++ extension: `pip install git+https://github.com/vietbt/VRPcpp.git`

## Code Structure 
    ├── env
    │   ├── Score.py            # Solution cost class
    │   ├── VRPNode.py          # Node class
    │   ├── InitSolution.py     # Clockhand/DBCA initializer
    │   ├── VRPInstance.py      # CC & IIL frameworks
    │   ├── VRPEnv.py           # Main env wrapper
    │   ├── VectorizedVRP.py    # Wrap to multi-processing env
    │   └── utils.py            # Environment utilities
    ├── network
    │   ├── encoder.py          # GatConv encoder
    │   ├── decoder.py          # Pointer Attention for k-opt
    │   ├── layers.py           # Neural network layers
    │   ├── model.py            # PPO main model
    │   └── policy.py           # Policy and IIL loss function
    ├── init_solution.py        # Initate solutions
    ├── runner.py               # Env runner with replay buffer
    ├── main.py                 # Train
    └── evaluate.py             # Evaluate
All 6 datasets indicated on our paper are available for downloading at <a href="https://drive.google.com/drive/u/0/folders/1qduTl6YBFRx6alrvijAxOXILDu-LEPdR">here</a>. 
You can download and decompress them to the current directory.

## Instructions 

### Solution Initialization (optional)
 - Run `python init_solution.py --data-folder=<data-folder>` for initializing solutions for all instances of the input directory and saving to `__pycache__`
### Training
 - Run `python main.py --seed=<seed> --data-folder=<data-folder> --algo=<algo> ...` for training deep RL and save all optimal solutions to `./logs` folder
 - We have many arguments for `main.py` such as which heuristic to be used, using imitation loss or not, round solution cost (similar to CVRPLIB authors), number of tours of sub-problem selection, number of workers, etc.
 - We ran the following command lines in our experiments:
    | ID | Method  | Command line |
    | ------- | ------  | ------------ |
    | 1 | IIL+VNS | `python main.py --data-folder=dataset/train/data_cvrp --algo=VNS --round-int --max-count=128` |
    | 1 | RL+VNS  | `python main.py --data-folder=dataset/train/data_cvrp --algo=VNS --round-int --max-count=128 --imitation-rate=0` |
    | 1 | IIL+HGS | `python main.py --data-folder=dataset/train/data_cvrp --algo=HGS --round-int --max-count=48` |
    | 1 | RL+HGS  | `python main.py --data-folder=dataset/train/data_cvrp --algo=HGS --round-int --max-count=48 --imitation-rate=0` |
    | 2 | IIL+HGS | `python main.py --data-folder=dataset/train/uniform_N500 --algo=HGS --max-count=4` |
    | 2 | IIL+HGS | `python main.py --data-folder=dataset/train/uniform_N2000 --algo=HGS --max-count=1` |
    | 2 | IIL+HGS | `python main.py --data-folder=dataset/train/uniform_N1000 --algo=HGS --max-count=2` |
    | 3 | IIL+VNS | `python main.py --data-folder=dataset/train/data_dimacs --algo=VNS --round-int --max-count=1024` |
    | 3 | RL+VNS  | `python main.py --data-folder=dataset/train/data_dimacs --algo=VNS --round-int --max-count=1024 --imitation-rate=0` |
    | 3 | IIL+HGS | `python main.py --data-folder=dataset/train/data_dimacs --algo=HGS --round-int --max-count=1024` |
    | 3 | RL+HGS  | `python main.py --data-folder=dataset/train/data_dimacs --algo=HGS --round-int --max-count=1024 --imitation-rate=0` |
    | 4 | IIL+VNS | `python main.py --data-folder=dataset/train/realworld/Leuven --algo=VNS --max-count=1024 --min-extend-nodes=512 --n-envs=16` |
    | 4 | IIL+HGS | `python main.py --data-folder=dataset/train/realworld/Leuven --algo=HGS --max-count=1024 --min-extend-nodes=512 --n-envs=16` |
    | 4 | IIL+VNS | `python main.py --data-folder=dataset/train/realworld/Ghent --algo=VNS --max-count=1024 --min-extend-nodes=512 --n-envs=16` |
    | 4 | IIL+HGS | `python main.py --data-folder=dataset/train/realworld/Ghent --algo=HGS --max-count=1024 --min-extend-nodes=512 --n-envs=16` |
    | 4 | IIL+VNS | `python main.py --data-folder=dataset/train/realworld/Flanders --algo=VNS --max-count=1024 --min-extend-nodes=1024 --n-envs=16` |
    | 4 | IIL+HGS | `python main.py --data-folder=dataset/train/realworld/Flanders --algo=HGS --max-count=1024 --min-extend-nodes=1024 --n-envs=16` |
    | 4 | IIL+VNS | `python main.py --data-folder=dataset/train/realworld/Brussels --algo=VNS --max-count=1024 --min-extend-nodes=1024 --n-envs=16` |
    | 4 | IIL+HGS | `python main.py --data-folder=dataset/train/realworld/Brussels --algo=HGS --max-count=1024 --min-extend-nodes=1024 --n-envs=16` |
    | 4 | IIL+VNS | `python main.py --data-folder=dataset/train/realworld/Antwerp --algo=VNS --max-count=1024 --min-extend-nodes=512 --n-envs=16` |
    | 4 | IIL+HGS | `python main.py --data-folder=dataset/train/realworld/Antwerp --algo=HGS --max-count=1024 --min-extend-nodes=512 --n-envs=16` |
    | 5 | IIL+VNS | `python main.py --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128` |
    | 5 | RL+VNS  | `python main.py --data-folder=dataset/train/data_evrp_wcci --algo=VNS --max-count=128 --imitation-rate=0` |
    | 5 | IIL+HGS | `python main.py --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24` |
    | 5 | RL+HGS  | `python main.py --data-folder=dataset/train/data_evrp_wcci --algo=HGS --max-count=24 --imitation-rate=0` |
- Add `--seed=<seed>` to each command line if you want more random seeds
- Due to the limitation of computing resources, we cannot benchmark all methods for each dataset
### Evaluating
 - Run `python evaluate.py --seed=<seed> --data-folder=<data-folder> --algo=<algo> --model-path=<trained-model-path> ...` for testing the generalizability of our models
 - We ran the following command lines in our experiments:
    | ID | Method  | Command line |
    | ------- | ------  | ------------ |
    | 2 | IIL+HGS| `python evaluate.py --data-folder=dataset/test/uniform_N500 --algo=HGS --max-count=1 --model-path=logs/dataset_train_uniform_N500_HGS_0.1/seed_1/model.pt` |
    | 2 | IIL+HGS| `python evaluate.py --data-folder=dataset/test/uniform_N1000 --algo=HGS --max-count=1 --model-path=logs/dataset_train_uniform_N1000_HGS_0.1/seed_1/model.pt` |
    | 2 | IIL+HGS| `python evaluate.py --data-folder=dataset/test/uniform_N2000 --algo=HGS --max-count=1 --model-path=logs/dataset_train_uniform_N2000_HGS_0.1/seed_1/model.pt` |
    | 6 | RL+VNS| `python evaluate.py --data-folder=dataset/test/data_evrp_random/seed_1 --algo=VNS --max-count=128 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.0/seed_1/model.pt` |
    | 6 | IIL+VNS| `python evaluate.py --data-folder=dataset/test/data_evrp_random/seed_1 --algo=VNS --max-count=128 --model-path=logs/dataset_train_data_evrp_wcci_VNS_0.1/seed_1/model.pt` |
    | 6 | RL+HGS| `python evaluate.py --data-folder=dataset/test/data_evrp_random/seed_1 --algo=HGS --max-count=24 --imitation-rate=0 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.0/seed_1/model.pt` |
    | 6 | IIL+HGS| `python evaluate.py --data-folder=dataset/test/data_evrp_random/seed_1 --algo=HGS --max-count=24 --model-path=logs/dataset_train_data_evrp_wcci_HGS_0.1/seed_1/model.pt` |
    | 6 | ...| ... |
 - Note that there are 14 different folders on dataset 6, which random generated with same distribution of dataset 5. Change `--data-folder` and `--model-path` respectively to evaluate all of them

## Data Analysis and Visualization
 - We create 6 notebooks for analyzing results of 6 datasets
 - Check all these files for more detail
    | ID | Command line |
    | ------- | ------------ |
    | 1  | `experiments_CVRP.ipynb` |
    | 2  | `experiments_CVRP_Uniform.ipynb` |
    | 3  | `experiments_DIMACS.ipynb` |
    | 4  | `experiments_realworld.ipynb` |
    | 5  | `experiments_EVRP.ipynb` |
    | 6  | `experiments_EVRP_generation.ipynb` |

## Contact
...
