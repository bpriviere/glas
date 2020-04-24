% reads binary file representing a set of piecewise polynomials - format given below.
% used to communicate with octomap_corridor function.
% output: cell array of matlab ppform structs
%
function pps = read_pptrajs(path)
%{
Header:
	N     - int32 - number of robots
	k     - int32 - number of timesteps
	dim   - int32 - dimension
	order - int32 - polynomial degree + 1
Breaks:
	t - float32[k + 1] - piece break times
Coefs:
	c - float32[order, dim, k, N] - first dimension contiguous - 
	                                ascending order (constant term first)
End:
	sentinel - 0xFFFFFFFF - for sanity check
%}
	endian = 'l'; % little-endian

	f = fopen(path, 'r');

	N = fread(f, 1, 'int32', 0, endian);
	K = fread(f, 1, 'int32', 0, endian);
	dim = fread(f, 1, 'int32', 0, endian);
	order = fread(f, 1, 'int32', 0, endian);
	breaks = fread(f, K+1, 'float32', 0, endian);

	pps = cell(N, 1);
	for i=1:N
		coefs = nan([dim K order]);
		for k=1:K
			for d=1:dim
				c = fread(f, order, 'float32', 0, endian);
				coefs(d,k,:) = flip(c);
			end
		end
		pps{i} = mkpp(breaks, coefs, dim);
	end

	sentinel = fread(f, 1, 'uint32', 0, endian);
	assert(sentinel == uint32(2^32 - 1));
	fclose(f);
end
