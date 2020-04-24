% permutation matrix to convert [xyz ... xyz]' into [x...x y...y z...z]'
% d: dimension (i.e. 3 for 3d points), k: number of points
function P = dim_collect_matrix(d, k)
	P = [];
	for i=1:d
		s = zeros(1,d);
		s(i) = 1;
		P = [P; kron(eye(k), s)];
	end
	assert(all(size(P) == k * d));
end
