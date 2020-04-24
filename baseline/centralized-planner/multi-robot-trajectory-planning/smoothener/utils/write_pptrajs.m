% writes binary file representing a set of piecewise polynomials - format given below.
% used to communicate with octomap_corridor function.
% input: {N} cell array of matlab ppform structs, [string] output file path
%
function write_pptrajs(pps, path)
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

	N = length(pps);
	[breaks, ~, K, order, dim] = unmkpp(pps{1});
	for i=2:N
		assert_pps_match(pps{1}, pps{i});
	end

	f = fopen(path, 'w');
	fwrite(f, N, 'int32', 0, endian);
	fwrite(f, K, 'int32', 0, endian);
	fwrite(f, dim, 'int32', 0, endian);
	fwrite(f, order, 'int32', 0, endian);

	fwrite(f, breaks, 'float32', 0, endian);

	for i=1:N
		coefs = pps{i}.coefs;
		coefs = reshape(coefs, [dim K order]);
		for k=1:K
			for d=1:dim
				c = squeeze(coefs(d,k,:))';
				c = flip(c);
				fwrite(f, c, 'float32', 0, endian);
			end
		end
	end

	fwrite(f, 2^32 - 1, 'uint32', 0, endian);
	fclose(f);
end
