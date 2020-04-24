% given N sequences of waypoints,
% creates N piecewise polynomial trajectories connecting the waypoints
% with piecewise-linear constant velocity segments
% and infinite acceleration at the waypoints.
%
% inputs:
%   paths:     [dim K+1 N] waypoints
%   timescale: [1] the amount of time each segment should take
%   order:     [1] the polynomial order
%
% output: {N} cell array of matlab ppform structs
%
function pps = path_linear_pps(paths, timescale, order)
	[dim, Kp1, N] = size(paths);
	K = Kp1 - 1;
	pps = cell(N, 1);
	breaks = timescale * (0:K);
	for i=1:N
		path = paths(:,:,i);
		coefs = zeros([dim K order]);
		slopes = diff(path, 1, 2) / timescale;
		coefs(:,:,end) = path(:,1:(end-1));
		coefs(:,:,(end-1)) = slopes;
		pps{i} = mkpp(breaks, coefs, dim);
	end
end
