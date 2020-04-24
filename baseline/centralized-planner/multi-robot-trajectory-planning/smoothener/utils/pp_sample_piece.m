% samples <npts> equally spaced points of the <ipiece>th piece of ppform struct <pp>
function pts = pp_sample_piece(pp, ipiece, npts)
	t0 = pp.breaks(ipiece);
	t1 = pp.breaks(ipiece + 1);
	dt = (t1 - t0) / (npts - 1);
	t = t0 + dt * (0:(npts - 1));
	assert(abs(t(end) - t1) < 10e-9);
	pts = ppval(pp, t);
end
