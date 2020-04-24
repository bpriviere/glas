% compute the matrix Q for a polynomial coefficent vector p
% such that p' Q p = integral(square(d'th derivative of p)).
% where coefs in p are given in descending (polyval) order.
%
% from "Polynomial trajectory planning for quadrotor flight",
%      C. Richter, A. Bry, N. Roy, ICRA 2013.
%
function Q = int_sqr_deriv_matrix(degree, deriv, T)
	r = deriv;
	order = degree+1;
	Q = nan(order);
	for i=0:degree
		for l=0:degree
			if i < r || l < r
				Q(i+1,l+1) = 0;
			else
				p = 1;
				for m=0:(r-1)
					p = p * (i - m) * (l - m);
				end
				x = 1 * p * T^(i + l - 2*r + 1) / (i + l - 2*r + 1);
				Q(i+1,l+1) = x;
			end
		end
	end
	Q = rot90(rot90(Q));
end
