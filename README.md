# GLAS Instructions
# https://arxiv.org/pdf/2002.11807.pdf
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