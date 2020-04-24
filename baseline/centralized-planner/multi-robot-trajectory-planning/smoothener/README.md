# smoothener-heterogeneous (Under development!!!)
Convert multi-robot waypoint sequences into smooth piecewise polynomial trajectories.

This repository contains an expanded Matlab implementation of the continuous trajectory optimization stage
of the algorithm described in:

> *Downwash-Aware Trajectory Planning for Large Quadcopter Teams*
>
> James A. Preiss, Wolfgang Hönig, Nora Ayanian, Gaurav S. Sukhatme
>
> Accepted at IEEE IROS 2017,
> Preprint at [https://arxiv.org/abs/1704.04852](https://arxiv.org/abs/1704.04852)

The extension allows for trajectory optimization of heterogenous agent sets.

NOTE: This is currently under development and does not fully function!!!


## overview

The purpose of this program is to convert a waypoint sequence for multiple robots
into a set of smooth trajectories.
It is assumed that the waypoint sequence comes from a planner (typically graph-based)
that models robots moving on straight lines between waypoints.
It is impossible to execute such a plan in real life with nonzero velocity at the waypoints,
because it would require infinite acceleration.
Therefore, we want to "smoothen" the trajectories while maintaining
the collision avoidance guarantee.

We model robots as axis-aligned ellipsoids to deal with the downwash effect of quadrotors.
The output format is piecewise polynomial trajectories with user-specified smoothness and degree.
Output is given as Matlab's [ppform](https://www.mathworks.com/help/curvefit/the-ppform.html) structs.


## setup instructions
1. Make sure your Matlab MEX compiler is set up correctly, and uses at least -O2 optimization.
2. Run `make`.
3. From the `smoothener` root directory , open a Matlab session and run `smoothener.m`.
   Computation should take several seconds, and you should see a 3D plot when it is done.
4. Make sure matlab is launched to use system libstdc++.so.6 rather than the Matlab distributed version.
   This can be done by using LD_PRELOAD
* alias matlab="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/local/MATLAB/R2017b/bin/matlab -desktop"

### extra setup to use Octomap:
4. run `git submodule init && git submodule update`.
5. `cd` into `octomap_corridor/octomap` and follow the "to only compile the library" instructions in [Octomap's `README.md`](https://github.com/OctoMap/octomap).
6. `cd` back into the `smoothener` root directory and run `make octomap`.
7. From the `smoothener` root directory , open a Matlab session and run `main_octomap`.
   Computation should take several seconds, and you should see a 3D plot when it is done.


## environment obstacles

Currently supported environment models are:

* [Octomap](https://octomap.github.io/)s, given as binary files on the disk
* sets of axis-aligned boxes

This part of the code is modularized, making it easy to add support for other environment representations.


## project structure

item | description
---- | -----------
`<top level>` | main routines, obstacle model implementations, and makefile. The main entry point is `smoothener.m`.
`examples/` | example problems for both "octomap" and "list of boxes" environment models, and an obstacle-free problem.
`external/` | third-party code from the Matlab file exchange.
`octomap_corridor/` | a standalone program that computes the safe corridor for a single robot in an octomap. A separate process is used instead of a mex-function because the auto-generated SVM solver C code from CVXGEN uses global variables, making multithreading impossible.
`svm_sephyp_*point/` | generated C code from the CVXGEN package to solve ellipsoid-weighted hard margin SVM problems for separating hyperplanes in 3D. Used for both robot-robot and robot-obstacle separation. Also contains mex-functions implementing the outer loop for all robots.
`tests/` | very few unit tests, need to write some more...
`utils/` | low-level simple subroutines.

   
## project status

### short-horizon TODOs:
- Support 2D problems. Most "difficult" parts of the code are dimension-generic, but some places assume 3D,
  particularly the CVXGEN generated code for separating hyperplane problems.
  In the meantime, use a discrete plan with all zeroes for the z-axis,
  and set the z bounding box to something like +/- 0.001.
- Support varying time allocation to each step in the discrete plan (each step is same duration for all robots, but not all steps in sequence need to be same duration)
- Fix the hard-coded ellipsoid dimensions in `octomap_corridor.cpp`.
- Write more unit tests.

### long-horizon TODOs:
- Port to Python, Julia, ... (**outside contributors welcomed!**)
- Port to C++

## implementation notes

The implementation uses generated code from
[CVXGEN](https://cvxgen.com/docs/index.html) to solve many small separating-hyperplane problems.
This is necessary to avoid excessive overhead of interpreting Matlab code.
The makefile compiles all C/C++ code.

The implementation has been profiled and optimized.
There is no known low-hanging fruit.
For medium-sized problems (20-80 robots), the bottleneck is solving the large quadratic programs
for each robot's corridor-constrained trajectory optimization problem.
If [CPLEX](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/) is installed,
it will be used to solve these QPs.

Wherever possible, computational bottlenecks are parallelized.
Multiple CPU cores up to the number of robots will show a large benefit.


## acknowledgements

This project is developed under the supervision of [Gaurav S. Sukhatme](http://www-robotics.usc.edu/~gaurav/)
as a part of the [Robotic Embedded Systems Lab (RESL)](http://robotics.usc.edu/resl/).

This work originated as a project in [Nora Ayanian](http://www-bcf.usc.edu/~ayanian/)'s course *Coordinated Mobile Robotics* at USC in Fall 2016.

Ongoing development has been a collaboration with [Wolfgang Hönig](https://github.com/whoenig) who provided the discrete graph planning front end and contributed to debugging and experiments.

The method builds upon [Sarah Tang](http://www.seas.upenn.edu/~sytang/)'s IROS 2016 [paper](http://www.seas.upenn.edu/~sytang/docs/2016IROS.pdf).

The project name is from [Anna Lukina](http://logic-cs.at/phd/students/anna-lukina/).
