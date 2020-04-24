#include <vector>
#include <Eigen/Core>

//TODO: prob shouldn't rely on macros
#define SMALL_NUM 0.000001
#define ABS(x) ((x) >= 0 ? (x) : -(x))   //  absolute value
#define MAX(x,y) ((x) > y ? (x) : (y))   //  max
#define MIN(x,y) ((x) < y ? (x) : (y))   //  min

//Unaligned Eigen types to use in stl containers
// TODO?: I think eigen has an stl specialization for their aligned types.
typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vec3f_t;

struct ConflictCylinder{
  //Conflict cylinder definition cyl(above, below, radius)
  //  Cylinder defined for each agent type pair
  //  For types 'A' and 'B' the cylinder parameters are read as:
  //    Agent 'A' is required to be cyl.above z distance above 'B'
  //    Agent 'A' is required to be cyl.below z distance below 'B'
  //    Agent 'A' is required to be cyl.radius xy radius away from 'B'
  ConflictCylinder(): above(0),below(0),radius(0){}
  ConflictCylinder(float safeAbove, float safeBelow, float radius)
    : above(safeAbove), below(safeBelow), radius(radius){}

  float above;
  float below;
  float radius;

  // getFCLcylinder(fcl::Cylinder& cyl, fcl::Vec3f& centerOffset){
  //   cyl = fcl::Cylinder(radius,above + below);
  //   float zOffset = above - (above + below)/2;
  //   centerOffset = fcl::Vec3f(0,0,zOffset);
  // }

};

class SweptCylinderHull{

public:

  SweptCylinderHull(){
    init();
  }

  //AABB, updated after every computeHull call
  float zmin;
  float zmax;
  float ymin;
  float ymax;
  float xmin;
  float xmax;

  //vertices, updated after every computeHull call
  std::vector<vec3f_t> allVerts;
  std::vector<vec3f_t> hullVerts;

  //computes hull, populates hullVerts, updates AABB
  void computeHull(const ConflictCylinder& cyl, const vec3f_t& from, const vec3f_t& to){
    //swept direction
    vec3f_t dir = to - from;
    
    //dirxy is used to orient polytope hull vertices for direction of travel.
    vec3f_t dirxy = dir; 
    dirxy.z() = 0;
    //use actual dir if moving on xy
    if (ABS(dirxy.x()) > SMALL_NUM || ABS(dirxy.y()) > SMALL_NUM){
      dirxy = dirxy / dirxy.norm();
    }
    else{ //otherwise vertical only or no movement, assume axis aligned box
      dirxy.x() = 1;
      dirxy.y() = 0;
    }
    
    //XY Direction pattern for generating hull vertices
    hullpat[0] = dirxy.transpose() * rot45;
    hullpat[1] = hullpat[0].transpose() * rot90;
    hullpat[2] = -1 * hullpat[0];
    hullpat[3] = -1 * hullpat[1]; 

    //bottom verts for from position
    for (int i = 0; i < 4; ++i){
      allVerts[i].x() = hullpat[i].x() * cyl.radius * 1.414214 + from.x();
      allVerts[i].y() = hullpat[i].y() * cyl.radius * 1.414214 + from.y();
      allVerts[i].z() = from.z() - cyl.below;
    }
    //top verts for from position
    for (int i = 4; i < 8; ++i){
      allVerts[i].x() = allVerts[i-4].x();
      allVerts[i].y() = allVerts[i-4].y();
      allVerts[i].z() = from.z() + cyl.above;
    }
    //top and botom verts for to position just translate from verts by dir
    for (int i = 8; i < 16; ++i){
      allVerts[i] = allVerts[i-8] + dir;
    }
  
    //Pick hull vertices from allVerts set
    //case - moving on x,y
    std::vector<int> hullinds;
    if ((ABS(dir.x()) > SMALL_NUM) || (ABS(dir.y()) > SMALL_NUM)){
      if (ABS(dir.z()) < SMALL_NUM){ //horizontal only
        hullinds = std::vector<int>{1,2,5,6,8,11,12,15};
      }
      else if(dir.z() > 0){ //+z diagonal
        hullinds = std::vector<int>{0,1,2,3,5,6,8,11,12,13,14,15};
      } 
      else if (dir.z() < 0){ //-z diagonal
        hullinds = std::vector<int>{1,2,4,5,6,7,8,9,10,11,12,15};
      }
    }
    //case - moving only on +z
    else if (dir.z() > 0){
      hullinds = std::vector<int>{0,1,2,3,12,13,14,15};
    }
    //case - moving only on -z
    else if (dir.z() < 0){
      hullinds = std::vector<int>{4,5,6,7,8,9,10,11};
    }
    else{ //zero length edge; 
      //vertices are AABB for cylinder at from location
      hullinds = std::vector<int>{0,1,2,3,4,5,6,7};
    }
    updateHullVerts(hullinds);
    updateAABB();
  }

private:

  //needed rotations
  Eigen::Matrix<float, 3, 3> rot45;
  Eigen::Matrix<float, 3, 3> rot90;

  //direction pattern for generating vertices
  std::vector<vec3f_t> hullpat;

  //Plane normals / dist
  // std::vector<vec3f_t> convex_A;
  // std::vector<float> convex_b;

  void updateHullVerts(const std::vector<int>& vertInds){
    hullVerts.clear();
    for (int ind : vertInds){
      hullVerts.push_back(allVerts[ind]);
    }
  }

  void updateAABB(){
    float fmin = std::numeric_limits<float>::min();
    float fmax = std::numeric_limits<float>::max();
    xmin = fmax; ymin = fmax; zmin = fmax;
    xmax = fmin; ymax = fmin; zmax = fmin;

    for (const auto& vert : hullVerts){
      xmin = MIN(vert.x(),xmin);
      ymin = MIN(vert.y(),ymin);
      zmin = MIN(vert.z(),zmin);
      xmax = MAX(vert.x(),xmax);
      ymax = MAX(vert.y(),ymax);
      zmax = MAX(vert.z(),zmax);
    }
  }

  void init(){
    //needed rotation matrices. 
    //   TODO: Could probably just hardcode the multiplications directly...
    rot45 << 0.707107, -0.707107, 0.0,
           0.707107,  0.707107, 0.0,
           0.0,       0.0 ,     0.0;
    rot90 << 0.0, -1.0, 0.0,
             1.0,  0.0, 0.0,
             0.0,  0.0, 0.0;
    //fixed container sizes
    hullpat.resize(4);
    allVerts.resize(16);

    //initialize aabb
    // float fmin = std::numeric_limits<float>::min();
    // float fmax = std::numeric_limits<float>::max();
    // xmin = fmax; ymin = fmax; zmin = fmax;
    // xmax = fmin; ymax = fmin; zmax = fmin;
  }
};