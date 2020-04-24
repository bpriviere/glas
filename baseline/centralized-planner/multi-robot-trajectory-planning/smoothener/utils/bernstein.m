% compute the Berenstein polynomials of degree k.
% returns a (k+1) x (k+1) matrix where each row is a polynomial
% descending from x^k coefficient to constant coefficient.
function B = bernstein(n)
	B = zeros(n+1);
	for k = 0:n
		for i=k:n
			B(k+1,i+1) = (-1)^(i-k) * nchoosek(n, i) * nchoosek(i, k);
		end
	end
	B = fliplr(B);
end
