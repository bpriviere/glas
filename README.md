# GLAS Instructions
1. Install the necessary dependencies with conda:
    ```bash
    conda env create --file environment.yml
    conda activate glas_env
    ```
2. Create Training Dataset (in ~/data)
    ```bash
	python runAll.py 
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