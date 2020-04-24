./discretePlanning.sh $1
if test -f "multi-robot-trajectory-planning/examples/ground/output/discreteSchedule.yaml"; then
	cd multi-robot-trajectory-planning/smoothener
	matlab -nosplash -nodesktop -r "path_setup,smoothener,quit"
	cd ../..
	./export.sh $1 $2
fi