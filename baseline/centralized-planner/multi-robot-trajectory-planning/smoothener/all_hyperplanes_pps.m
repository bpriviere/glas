% given the input of all robots' polynomial trajectories,
% computes separating hyperplanes for all robot-robot pairs and all pp pieces.
%
% inputs: pps: {NROBOTS} cell array of Matlab ppform structs
%         ellipsoid: [3] ellipsoid radii
%
% outputs: A: [DIM x NROBOTS x NROBOTS x (NPTS - 1)] array of 
%             hyperplane normal vectors for each robot-robot interaction
%             at each segment. Will always be unit vectors.
%          b: distance from origin for hyperplanes. i.e. a(:,...)^T x <= b.
%
function [A, b] = all_hyperplanes_pps(pps, ellipsoid)

	NPTS_EVAL = 32;

	N = length(pps);
	DIM = pps{1}.dim;
	steps = pps{1}.pieces;
	breaks = pps{1}.breaks;
	for i=2:N
		assert(pps{i}.dim == DIM);
		assert(pps{i}.pieces == steps);
		assert(all(abs(pps{i}.breaks - breaks) < 10e-9));
	end

	paths = nan(DIM, NPTS_EVAL, N, steps);
	for step=1:steps
		for i=1:N
			paths(:,:,i,step) = pp_sample_piece(pps{i}, step, NPTS_EVAL);
		end
	end

	% separating hyperplanes for each robot-robot pair
	A = nan([DIM N N steps]);
	b = nan([N N steps]);

	parfor step=1:steps
		step_paths = paths(:,:,:,step);
		[Aslice, bslice] = all_hyperplanes_pps_mex(step_paths, ellipsoid);
		A(:,:,:,step) = Aslice;
		b(:,:,step) = bslice;
    end
end

