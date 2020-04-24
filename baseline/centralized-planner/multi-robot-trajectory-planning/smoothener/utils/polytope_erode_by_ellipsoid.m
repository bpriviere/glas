% A, b: half-space representation of polytope, {x : Ax <= b}
% ellipsoid: diagonal matrix D of axis-aligned ellipsoid radii
%
% output: new right-hand-side b' for the same A matrix such that,
%         if Ax <= b', then the ellipsoid of given radii centered at x 
%         lies inside the original polytope Ax <= b
%
function b = polytope_erode_by_ellipsoid(A, b, ellipsoid)
	[nr, nc] = size(A);
	for i=1:nr
		normal = A(i,:);
		if ~any(isnan(normal))
			assert(abs(norm(normal) - 1) < 0.00001);
			b(i) = b(i) - norm(ellipsoid * normal');
		end
	end
end
