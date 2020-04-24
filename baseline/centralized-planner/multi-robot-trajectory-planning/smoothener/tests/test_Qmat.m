function test_qmat()
	trials = 100;
	for i=1:trials
		deg = 4 + floor(rand() * 6);
		p = rand(1,deg+1);

		T = 2*rand() + 1;
		assert(T >= 1);
		Q = int_sqr_deriv_matrix(deg, 2, T);

		x1 = p * Q * p';
		x2 = non_matrix_version(p, T);

		assert(abs(x1 - x2) < 0.001);
	end
end

function i = non_matrix_version(p, T)
	d2p = polyder(polyder(p));
	dsqr = conv(d2p, d2p);
	intsqr = polyint(dsqr);
	i = polyval(intsqr, T);
end
