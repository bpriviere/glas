% inputs
%     p: polynomial coefs, descending, i.e. a polyval input
%     s: time scale, e.g. if s==2 the new polynomial will take 2x longer
%
% output:
%     stretched polynomial
%
function p = polystretchtime(p, s)
	p = flip(p);
	recip = 1.0 / s;
	scale = recip;
	order = length(p);
	for i=2:order
		p(i) = scale * p(i);
		scale = scale * recip;
	end
	p = flip(p);
end
