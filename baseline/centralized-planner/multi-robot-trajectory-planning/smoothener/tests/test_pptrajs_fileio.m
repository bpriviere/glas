function test_pptrajs_fileio()
	L = load('drones32b.mat');
	pps = L.pps(end,:);
	fname = 'temp.pptrajs';
	write_pptrajs(pps, fname);
	pps2 = read_pptrajs(fname);
	assert(length(pps) == length(pps2));
	N = length(pps);
	for i=1:N
		pp1 = pps{i};
		pp2 = pps2{i};
		assert_pps_match(pp1, pp2);
        assert(all(single(pp1.coefs(:)) == single(pp2.coefs(:))));
	end
end
