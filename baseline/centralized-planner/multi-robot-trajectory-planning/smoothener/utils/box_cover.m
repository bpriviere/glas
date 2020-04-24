% greedy box cover for voxel obstacles. output boxes can overlap.
% input:  3xN list of obstacle voxel centers, assume on integers and 1-unit side
% output: 3x2xK list of boxes - boxes(:,:,i) is [corner1 corner2]
%
function boxes = box_cover(cubes)

	if isempty(cubes)
		boxes = [];
		return;
	end

	% convert to voxels
	low = min(cubes, [], 2);
	cubes = bsxfun(@minus, cubes, low - 1);
	dims = max(cubes, [], 2)';
	voxels = zeros(dims);
	for i=1:size(cubes,2)
		voxels(cubes(1,i), cubes(2,i), cubes(3,i)) = 1;
	end
	uncovered = voxels;

	boxes = [];

	while any(uncovered(:))
		[x,y,z] = ind2sub(dims, find(uncovered, 1));
		box = [x x; y y; z z];
		ok = true;
		while ok
			ok = false;
			for d=1:3
				if box(d,2) < dims(d)
					box(d,2) = box(d,2) + 1;
					if all(voxels(box(1,1):box(1,2),box(2,1):box(2,2),box(3,1):box(3,2)))
						ok = true;
					else
						box(d,2) = box(d,2) - 1;
					end
				end
			end
		end
		boxes = cat(3, boxes, box);
		uncovered(box(1,1):box(1,2),box(2,1):box(2,2),box(3,1):box(3,2)) = 0;
	end

	boxes = bsxfun(@plus, boxes, low - 1);
end
