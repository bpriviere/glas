% dummy function that returns empty robot-obstacle halfspace constraints
% for problems with no obstacles.
%
% inputs:
%   pps: {N} cell array of matlab ppform structs
%   obs_ellipsoid: [3] radii of ellipsoid for robot/obstacle collision model
%
% outputs:
%   polytopes: {N k-1} cell array of 0x4 matrices representing no constraints
%
function polytopes = pp_obs_sep_none(pps, obs_ellipsoid)

	N = length(pps);
	k = length(pps{1}.breaks);
	polytopes = repmat({zeros(0,4)}, N, k-1);
end
