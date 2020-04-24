% computes the robot-obstacle halfspace constraints
% for the "octomap" obstacle model (using the Octomap C++ library)
%
% inputs:
%   pps: {N} cell array of matlab ppform structs
%   obs_ellipsoid: [3] radii of ellipsoid for robot/obstacle collision model
%   octomap_filepath: [string] file path of octomap, should have .bt extension.
%
% outputs:
%   polytopes: {N k-1} cell array of halfspaces
%              such that p = polytopes{irobot, istep} is a [4 nfaces] array
%              describing the linear system p(:,1:3)x <= p(:,4)
%
function polytopes = pp_obs_sep_octomap(pps, obs_ellipsoids, octomap_filepath)

	N = length(pps);
	k = length(pps{1}.breaks);

	polytopes = cell(N,k-1);
    
    ellip_x = obs_ellipsoids(:,1);
    ellip_y = obs_ellipsoids(:,2);
    ellip_z = obs_ellipsoids(:,3);

	% parfor
	parfor n=1:N
		pp_filepath = tempname();
		hs_filepath = tempname();
		write_pptrajs(pps(n), pp_filepath);
		cmd = sprintf('octomap_corridor/octomap_corridor %s %s %s %f %f %f', ...
			octomap_filepath, pp_filepath, hs_filepath,...
            ellip_x(n), ellip_y(n), ellip_z(n));
		status = system(cmd);
		assert(status == 0);
		hs_slice = read_halfspaces(hs_filepath);
		assert(all(size(hs_slice) == [1 k-1]));
		polytopes(n,:) = hs_slice;
		delete(pp_filepath);
		delete(hs_filepath);
	end
end
