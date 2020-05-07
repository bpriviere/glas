# GLAS Instructions

The repository contains the code for "GLAS: Global-to-Local Safe Autonomy Synthesis for Multi-Robot Motion Planning with End-to-End Learning", Benjamin Riviere, Wolfgang Hoenig, Yisong Yue, and Soon-Jo Chung, to appear at IEEE Robotics and Automation Letters (RA-L), 2020. A pre-print of the paper is available at https://arxiv.org/pdf/2002.11807.pdf

## Training Data Preparation

The data is available in a shared Box folder: https://caltech.box.com/s/doc8hszo55m4x0oyeo1ok2yxh1m2xq3z. The dataset is split into parts with 10000 example instances each. The content of the whole dataset was shuffled before splitting it into parts, i.e., for a smaller dataset it suffices to download only a subset of the datasets.


1. Extract the data into the data subfolder. For example, if you download and extract `training/doubleintegrator/part1.7z`, you should have the file `data/training/doubleintegrator/instances/map_8by8_obst06_agents004_ex000007.yaml`.

2. As verification step, visualize one of the examples:

```
python3 utils/plot_data.py data/training/doubleintegrator/instances/map_8by8_obst06_agents004_ex000007.yaml data/training/doubleintegrator/central/map_8by8_obst06_agents004_ex000007.npy
```

![Alt text](/docs/plot_di_map_8by8_obst06_agents004_ex000007.png?raw=true "Example plot")

Note that the example instance files are regular yaml files, containing a list of start locations, goal locations, and obstacles. The npy files are numpy matrices containing the sampled trajectory for each of the robots (see `plot_data.py` to understand the matrix layout).


## Usage 

NOTE: This section is still work in progress!


1. Install the necessary dependencies with conda:
    ```bash
    conda env create --file environment.yml
    conda activate glas_env
    ```
2. create training dataset 
    ```bash
    # make instance files (in ~/data/singleintegrator/instances)
    python ../../baseline/centralized-planner/genRandomMAPF.py
    # run central planner solution (in ~/data/singleintegrator/)
	python runAll.py --central 
    ```
3. train policy with Imitation Learning (in ~/results)
    ```bash
	python3 singleintegrator/exp1.py --train
    ```
4. simulate policy on test cases (in ~/results)
    ```bash
	python3 singleintegrator/exp1.py --sim
    ```
5. plot results (in ~/results)
    ```bash
	python3 singleintegrator/exp1.py --plot
    ```