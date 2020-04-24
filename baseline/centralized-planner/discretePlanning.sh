# rm old files
rm test.yaml
rm multi-robot-trajectory-planning/examples/ground/output/discreteSchedule.yaml
rm multi-robot-trajectory-planning/examples/ground/map.bt
rm multi-robot-trajectory-planning/examples/ground/output/map-ascii.stl
rm multi-robot-trajectory-planning/examples/ground/output/map.stl

# discrete planning
python3 discretePreProcessing.py $1 test.yaml
timeout 1m ./multi-robot-trajectory-planning/build/libMultiRobotPlanning/ecbs -i test.yaml -o multi-robot-trajectory-planning/examples/ground/output/discreteSchedule.yaml -w 1.3
# postprocess output (split paths)
python3 discretePostProcessing.py test.yaml multi-robot-trajectory-planning/examples/ground/output/discreteSchedule.yaml multi-robot-trajectory-planning/examples/ground/output/discreteSchedule.yaml
# python3 multi-robot-trajectory-planning/libMultiRobotPlanning/example/visualize.py $1 multi-robot-trajectory-planning/examples/ground/output/discreteSchedule.yaml
# convert yaml map -> octomap
./multi-robot-trajectory-planning/build/tools/map2octomap/map2octomap -m $1 -o multi-robot-trajectory-planning/examples/ground/map.bt
# convert octomap -> STL (for visualization)
./multi-robot-trajectory-planning/build/tools/octomap2openscad/octomap2openscad -i multi-robot-trajectory-planning/examples/ground/map.bt -o multi-robot-trajectory-planning/examples/ground/output/map.scad
openscad -o multi-robot-trajectory-planning/examples/ground/output/map-ascii.stl multi-robot-trajectory-planning/examples/ground/output/map.scad
stl2bin multi-robot-trajectory-planning/examples/ground/output/map-ascii.stl multi-robot-trajectory-planning/examples/ground/output/map.stl