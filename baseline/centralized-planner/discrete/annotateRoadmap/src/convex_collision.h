#pragma once
#include <vector>
// eigen
#include <Eigen/Core>
typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vec3f;

// takes two point clouds as input and computes the convex hull for each pointcloud
// returns true, if those two convex hulls intersect
bool convex_collision(
  const std::vector<vec3f>& convexObj1,
  const std::vector<vec3f>& convexObj2,
  float mpr_tolerance  = 0.0001);

#ifdef ENABLE_CONVEX_HULL_STL_OUTPUT
void convex_hull_to_stl(
  const std::vector<vec3f>& convexObj,
  std::ostream& output);
#endif // ENABLE_CONVEX_HULL_STL_OUTPUT
