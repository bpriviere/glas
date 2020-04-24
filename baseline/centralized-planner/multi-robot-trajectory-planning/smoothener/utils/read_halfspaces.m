% reads binary file representing safe corridor polytopes for N robots over k timesteps.
% used to communicate with octomap_corridor function.
% file format described below.
%
% output: {N k} cell array of [4 nfaces] halfspace inequalities
%
function hs = read_halfspaces(path)
	endian = 'l'; % little-endian

	%{
	.halfspaces format:
	Header:
		N     - int32 - number of robots
		k     - int32 - number of timesteps
		dim   - int32 - dimension
	for N robots:
		for K timesteps:
			M - int32 - number of halfspaces
			for M halfspaces:
				a - float32[dim] for ax <= b
				b - float32
	sentinel - 0xFFFFFFFF - for sanity check

	note: M can be different for each robot-timestep pair!
	%}

	f = fopen(path, 'r');

	N = fread(f, 1, 'int32', 0, endian);
	K = fread(f, 1, 'int32', 0, endian);
	dim = fread(f, 1, 'int32', 0, endian);
	assert(dim == 3);

	hs = cell(N, K);

	for n=1:N
		for k=1:K
			M = fread(f, 1, 'int32', 0, endian);
			step_hs = fread(f, [4 M], 'float32', 0, endian)';
			hs{n,k} = step_hs;
		end
	end

	sentinel = fread(f, 1, 'uint32', 0, endian);
	assert(sentinel == uint32(2^32 - 1));
	fclose(f);
end
