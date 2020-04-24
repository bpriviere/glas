% given a waypoint plan, create a new waypoint plan
% with a waypoint added at the middle of each segment.
%
% input:  [3 k N] waypoints
% output: [3 (2k - 1) N] waypoints
%
function r = refine_schedule(s)
	[~, k, N] = size(s);
	k2 = 2 * k - 1;
	r = zeros(3, k2, N);
	r(:,1,:) = s(:,1,:);
	delta = diff(s, [], 2) ./ 2;
	r(:,1:2:k2,:) = s;
	r(:,2:2:(k2-1),:) = s(:,1:(k-1),:) + delta;
end
