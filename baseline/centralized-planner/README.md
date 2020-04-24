# Centralized Planner for Multi-Robot Trajectory Planning

Based on

W. Hönig, J. A. Preiss, T. K. S. Kumar, G. S. Sukhatme, and N. Ayanian. "Trajectory Planning for Quadrotor Swarms", in IEEE Transactions on Robotics (T-RO), Special Issue Aerial Swarm Robotics, vol. 34, no. 4, pp. 856-869, August 2018. 

and

M. Debord, W. Hönig, and N. Ayanian. "Trajectory Planning for Heterogeneous Robot Teams", in Proc. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, Spain, October 2018. 

## Setup


See multi-robot-trajectory-planning Readme

```
sudo pip3 install numpy-stl
```

## Example

## Example

### Run ORCA Baseline

```
../orca/build/orca -i examples/test_2_agents.yaml
python3 ../orca/orca.py examples/test_2_agents.yaml orca.csv
```

### Run Planner

```
./run.sh examples/test_2_agents.yaml data/test_2_agents.npy
```

### Discrete Planning

````
./discretePlanning.sh examples/test_2_agents.yaml
````

### Continuous Optimization

```
(open matlab in multi-robot-trajectory-planning/smoothener)
(run path_config)
(run smoother)
```

### Temporal stretching and export

```
./export.sh examples/test_2_agents.yaml data/test_2_agents.npy
```

### Visualize

```
python3 ../orca/orca.py examples/empty-8-8-random-1_20_agents.yaml central.csv
```