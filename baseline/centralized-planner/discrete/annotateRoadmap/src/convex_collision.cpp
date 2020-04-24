#include "convex_collision.h"

#include <ccd/ccd.h>
// #include <Eigen/Dense>

vec3f computeCenter(const std::vector<vec3f>& points)
{
  vec3f result(0, 0, 0);
  for (const vec3f& p : points) {
    result += p;
  }
  return result / points.size();
}

struct ccd_convex_t
{
  ccd_convex_t(const std::vector<vec3f>* points)
    : points(points)
  {
    center = computeCenter(*points);
  }

  const std::vector<vec3f>* points;
  vec3f center;
};

/** Support function for convex object
    returns furthest point from object (shape) in specified direction.
**/
void support(const void* obj, const ccd_vec3_t* dir, ccd_vec3_t* vec)
{
  const ccd_convex_t* c = (const ccd_convex_t*)obj;
  const auto& center = c->center;
  Eigen::Vector3f dirVec(dir->v[0], dir->v[1], dir->v[2]);

  ccd_real_t maxdot = -CCD_REAL_MAX;

  for (const vec3f& p : *(c->points)) {
    float dot = dirVec.dot(p - center);
    if (dot > maxdot) {
      ccdVec3Set(vec, p[0], p[1], p[2]);
      maxdot = dot;
    }
  }
}

/** Center function - returns center of object */
void center(const void* obj, ccd_vec3_t* center)
{
  const ccd_convex_t* c = (const ccd_convex_t*)obj;
  ccdVec3Set(center, c->center[0], c->center[1], c->center[2]);
}

bool convex_collision(
  const std::vector<vec3f>& convexObj1,
  const std::vector<vec3f>& convexObj2,
  float mpr_tolerance)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  ccd_convex_t obj1(&convexObj1);
  ccd_convex_t obj2(&convexObj2);

  // set up ccd_t struct
  ccd.support1       = support;       // support function for first object
  ccd.support2       = support;       // support function for second object
  ccd.center1        = center;        // center function for first object
  ccd.center2        = center;        // center function for second object
  ccd.mpr_tolerance  = mpr_tolerance; // maximal tolerance

  int intersect = ccdMPRIntersect(&obj1, &obj2, &ccd);
  return intersect;
}

#ifdef ENABLE_CONVEX_HULL_STL_OUTPUT

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_hull_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef CGAL::Polyhedron_3<K>                     Polyhedron;
typedef K::Point_3                                Point;
typedef K::Vector_3                               Vector;

// see https://github.com/CGAL/cgal/blob/master/Polyhedron_IO/examples/Polyhedron_IO/off2stl.cpp
void polyhedron2STL(const Polyhedron& P, std::ostream& output)
{
  output << "solid " << std::endl;

  // write triangles
  for ( auto i = P.facets_begin(); i != P.facets_end(); ++i) {
    auto h = i->halfedge();
    if ( h->next()->next()->next() != h) {
      throw std::runtime_error("polyhedron is not triangulated");
    }
    Point p = h->vertex()->point();
    Point q = h->next()->vertex()->point();
    Point r = h->next()->next()->vertex()->point();
    // compute normal
    Vector n = CGAL::cross_product( q-p, r-p);
    Vector norm = n / std::sqrt( n * n);
    output << "    facet normal " << norm << std::endl;
    output << "      outer loop " << std::endl;
    output << "        vertex " << p << std::endl;
    output << "        vertex " << q << std::endl;
    output << "        vertex " << r << std::endl;
    output << "      endloop " << std::endl;
    output << "    endfacet " << std::endl;
  }

  output << "endsolid " << std::endl;
}

void convex_hull_to_stl(
  const std::vector<vec3f>& convexObj,
  std::ostream& output)
{
  std::vector<Point> points;
  points.reserve(convexObj.size());
  for (const auto& p : convexObj) {
    points.push_back(Point(p[0], p[1], p[2]));
  }

  // define polyhedron to hold convex hull
  Polyhedron poly;

  // compute convex hull of non-collinear points
  CGAL::convex_hull_3(points.begin(), points.end(), poly);

  polyhedron2STL(poly, output);
}

#endif // ENABLE_CONVEX_HULL_STL_OUTPUT
