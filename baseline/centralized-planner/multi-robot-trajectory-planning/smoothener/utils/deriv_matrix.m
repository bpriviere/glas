% returns the matrix M such that, if x is coefficients of 
% a k-degree polynomial in descending order (a polyval input),
% M * x is the coefficients of x's derivative.
function D = deriv_matrix(k)
	s = k:(-1):0;
	D = diag(s);
	D(2:end,:) = D(1:(end-1),:);
	D(1,:) = 0;
end
