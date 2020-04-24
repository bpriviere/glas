% read a map from Wolfgang's discrete planner
% dims: room box dimensions - grid cells 0 thru dims, inclusive, are valid
% obstacles: 3xN list of obstacle grid cells. 1unit cubes surround
%
function [dims, obstacles] = read_map(fname)
	json = loadjson(fname);
	dims = json.dimensions;
	obstacles = json.obstacles';
	if isempty(obstacles)
		obstacles = [];
	end
end
