# Discrete Planners with downwash awareness for quadcopters

```
sudo apt install libfcl-0.5-dev ros-kinetic-ompl ros-kinetic-octomap
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

Uses boost, fcl, ccd, octomap, ompl

## Generate Roadmap

```
../../build/discrete/generateRoadmap/generateRoadmap -e map.bt -r "<Cylinder:0.25,0.45>" -o output/roadmapTypeGround.yaml -a addVerticesTypeGround.yaml -c roadmapConfigTypeGround.yaml --type SPARS --dimension 2 --fixedZ 0.23
```