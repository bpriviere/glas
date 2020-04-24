% given the lower and upper bounding corners of a box,
% returns an array of all the box vertices.
%
% inputs:
%   x0: [3] lower corner of box.
%   x1: [3] upper corner of box.
%
% outputs:
%   verts: [8 3] all box corners.
%
function verts = corners_to_box_vertices(x0, x1)
	verts = [...
		x0(1) x0(2) x0(3);
		x0(1) x0(2) x1(3);
		x0(1) x1(2) x0(3);
		x0(1) x1(2) x1(3);
		x1(1) x0(2) x0(3);
		x1(1) x0(2) x1(3);
		x1(1) x1(2) x0(3);
		x1(1) x1(2) x1(3)];
end
