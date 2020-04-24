% computes the robot-obstacle halfspace constraints
% for the "list of boxes" obstacle model.
%
% inputs:
%   pps: {N} cell array of matlab ppform structs
%   obs_ellipsoid: [3] radii of ellipsoid for robot/obstacle collision model
%   boxes: [3 2 nObs] min/max corners of obstacle boxes
%
% outputs:
%   polytopes: {N k-1} cell array of halfspaces
%              such that p = polytopes{irobot, istep} is a [4 nfaces] array
%              describing the linear system p(:,1:3)x <= p(:,4)
%
function polytopes = pp_obs_sep_grid(pps, obs_ellipsoid, boxes)

	N = length(pps);
	k = length(pps{1}.breaks);
	[~, ~, nObs] = size(boxes);

	polytopes = cell(N,k-1);

	% parfor
	parfor irobot=1:N
		for istep = 1:(k-1)
			seg = pp_sample_piece(pps{irobot}, istep, 32)';
			polytope = zeros(nObs, 4);
			for iobs = 1:nObs
				[aa, bb] = path_box_hyperplane(...
					seg, boxes(:,1,iobs), boxes(:,2,iobs), obs_ellipsoid);
				polytope(iobs,1:3) = aa;
				polytope(iobs,4) = bb;
			end
			polytopes{irobot,istep} = polytope;
		end
	end
end

% TODO: if more performance is needed,
% rewrite the outer loop as a mex-function to avoid overhead
% of setting up the many CVXGEN problems in Matlab interpreted code.

% inputs:
%         path [2x3] or [32x3] path (segment or sampled pp)
%         x0[3] x1[3] box diagonally opposed corners
%         ell[3] ellipsoid radii
%
% outputs: [a, b] normalized hyperplane s.t. |a| = 1, a path < b, a cube > b
%
% note: this hyperplane is tight to the box and shifted by the ellipsoid
%
function [a, b] = path_box_hyperplane(path, x0, x1, ell)

	cube_pts = corners_to_box_vertices(x0, x1);
	cube_pts = repmat(cube_pts, 4, 1);

	if size(path, 1) == 2
		path = repmat(path, 16, 1);
	end

	assert(size(path, 1) == 32);

	[a, b] = separating_hyperplane(path, cube_pts, ell);

	% now we have a max-margin separating hyperplane shifted by ellipsoid
	% but we want a hyperplane tight to the cube, shifted by ellipsoid:
	min_dist = min(cube_pts * a - b);
	%ell_dist = norm(diag(ell) * a);
	b = b + min_dist;% - ell_dist;
end

% inputs: p1 [N * dim] path 1
%         p2 [M * dim] path 2
%         ell [3] ellipsoid radii
% output: [a, b] normalized hyperplane s.t. |a| = 1, a p1 < b, a p2 > b
%
function [a, b] = separating_hyperplane(p1, p2, ell)

	[N, dim] = size(p1);
	[M, dim2] = size(p2);
	assert(dim == dim2);

	A = [ p1  ones(N, 1); ...
	     -p2 -ones(M, 1) ];
	Q = diag([ell.^2 0]);

	params.A = A;
	params.Q = Q;
	settings.verbose = 0;

	assert(dim == 3 && N == M && N == 32);
	[vars, status] = svm32_mex(params, settings);
	beta = vars.beta;
	assert(length(beta) == dim + 1);

	a = beta(1:dim);
	b = -beta(end);
	na = norm(a);
	a = a ./ na;
	b = b ./ na;
end
