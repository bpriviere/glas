% input: [dim, nsteps, nrobots] discrete plan
function analyze_schedule(paths)
	[dim, steps, nrobots] = size(paths);
	fprintf('%d robots, %d timesteps\n', nrobots, steps);
	waittot = 0;
	for r=1:nrobots
		delta = diff(paths(:,:,r), [], 2);
		stationary = all(delta == 0, 1);
		waittot = waittot + sum(stationary);
	end
	fprintf('%d / %d steps are waits\n', waittot, steps * nrobots);
	zmin = Inf;
	for s=1:steps
		xypos = squeeze(paths(1:2,s,:));
		d = squareform(pdist(xypos'));
		d = d + diag(nan(1,nrobots));
		[row, col] = find(d == 0);
		for i=1:length(row)
			zdist = abs(paths(3,s,row(i)) - paths(3,s,col(i)));
			zmin = min(zmin, zdist);
		end
	end
	fprintf('%d minimum stacked z distance\n', zmin);
	allxyz = reshape(paths, dim, []);
	lb = min(allxyz, [], 2);
	ub = max(allxyz, [], 2);
	fprintf('bounding box: \n\tmin = [%2.3f %2.3f %2.3f]\n\tmax = [%2.3f %2.3f %2.3f]\n', ...
		lb(1), lb(2), lb(3), ub(1), ub(2), ub(3));
end
