smoothener:
	matlab -nosplash -nodesktop -nojvm -r "mex_required,quit"
	make -C octomap_corridor

clean:
	make clean -C octomap_corridor
	find . -name "*.mexa64" -type f -delete
