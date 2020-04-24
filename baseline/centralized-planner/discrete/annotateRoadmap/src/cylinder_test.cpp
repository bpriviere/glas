#include "ConflictCylinder.h"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {

	//Swept hull generator.
	SweptCylinderHull sweptHull;

	//Conflict cylinder definition cyl(safe above, safe below, safe radius)
	//	Cylinder defined for each agent type pair
	//	For types 'A' and 'B' the cylinder parameters are read as:
	//		Agent 'B' is required to be cyl.above z distance above 'A'
	//		Agent 'B' is required to be cyl.below z distance below 'A'
	//		Agent 'B' is required to be cyl.radius xy distance away from 'A'
	ConflictCylinder conflictCylinder(4,3,2);

	//Swept segment definition
	vec3f_t from;
	from << -2,2,1; //x,y,z
	vec3f_t to;
	to << 3,5,-2;

	//compute hull
	sweptHull.computeHull(conflictCylinder, from, to);

	//Hull vertices
	std::cout << "Hull Vertices:" << std::setprecision(3) << std::fixed << std::endl;
	for (auto &vert : sweptHull.hullVerts){
		std::cout << "  " << vert.transpose() << std::endl;
	}

	//Axis aligned bounding box parameters
	std::cout << std::endl << "AABB [min,max]:" << std::endl;
	std::cout << "  x: [" << sweptHull.xmin << ", " << sweptHull.xmax << "]" << std::endl;
	std::cout << "  y: [" << sweptHull.ymin << ", " << sweptHull.ymax << "]" << std::endl;
	std::cout << "  z: [" << sweptHull.zmin << ", " << sweptHull.zmax << "]" << std::endl;

	return 0;
}