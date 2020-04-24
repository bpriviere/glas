% writes a matlab ppform struct to a CSV file.
% TODO: document CSV format used.
%
function pp2csv(pp, filename)
	[breaks, coefs, npieces, order, dim] = unmkpp(pp);
	assert(dim == 3 || dim == 4);
	coefs = reshape(coefs, [], npieces, order);
	if dim == 3
		coefs = cat(1, coefs, zeros(1, npieces, order));
	end

	fid = fopen(filename, 'w');
	format long;
	vars = {'x' 'y' 'z' 'yaw'};
	fprintf(fid, 'duration,');
	for d=1:4
		for i=1:order
			fprintf(fid, '%s^%d,', vars{d}, i-1);
		end
	end
	fprintf(fid, '\n');
	for piece=1:npieces
		duration = breaks(piece+1) - breaks(piece);
		fprintf(fid, '%f,', duration);
		for d=1:4
			fprintf(fid, '%f,', flipud(squeeze(coefs(d,piece,:))));
		end
		fprintf(fid, '\n');
	end
end
