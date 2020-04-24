#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <thread>

// Boost
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

// Collision code
#include "cylinderAnnotation.h"
#include "convex_collision.h"

//timer
#include "StopWatch.hpp"

typedef Eigen::Matrix<float, 2, 1, Eigen::DontAlign> vec2f_t;
bool vv_conflict(const ConflictCylinder& cyl, const vec3f_t& cylPos, const vec3f_t& vPos);

// Assumes conflict cylinder is moving from edgeFromA to edgeToA.
// Check if the swept convex hull intersects with line segment edgeFromB to edgeToB
bool ee_conflict(const ConflictCylinder& cyl, const vec3f_t& edgeFromA, const vec3f_t& edgeToA, const vec3f_t& edgeFromB, const vec3f_t& edgeToB)
{
  SweptCylinderHull sweptHullA;
  sweptHullA.computeHull(cyl, edgeFromA, edgeToA);

  std::vector<vec3f> convexObj2;
  convexObj2.push_back(edgeFromB);
  convexObj2.push_back(edgeToB);

  return convex_collision(sweptHullA.hullVerts, convexObj2);
}

size_t omp_ee_conflicts(const ConflictCylinder& cyl, const AgentType& typeA, const AgentType& typeB, AnnotatedRoadmap& g, size_t numJobs)
{

  //edge-edge
  size_t nConflicts = 0;
  #pragma omp parallel for num_threads(numJobs) reduction(+:nConflicts)
  for (size_t i = typeA.eStart; i < typeA.eEnd; ++i){ //agentA on edge i
    for (size_t j = typeB.eStart; j < typeB.eEnd; ++j){ //agentB on edge j
      if (i == j){
        continue;
      }
      //check if cylinder moving on 'i' intersects with edge 'j'
      if( ee_conflict(cyl,
          g.vertices[g.edges[i].from].pos, g.vertices[g.edges[i].to].pos,
          g.vertices[g.edges[j].from].pos, g.vertices[g.edges[j].to].pos)) {
        g.edges[i].edgeCollisions[0].insert(j);
        nConflicts += 1;
      }

    }
  }
  return nConflicts;
}

// http://geomalgorithms.com/a05-_intersect-1.html
int intersect3D_SegmentPlane(
  const vec3f_t& edgeFrom,
  const vec3f_t& edgeTo,
  const vec3f_t& planePt,
  const vec3f_t& planeNormal,
  vec3f_t* result)
{
  vec3f_t u = edgeTo - edgeFrom;
  vec3f_t w = edgeFrom - planePt;

    float D = planeNormal.dot(u);
    float N = -planeNormal.dot(w);

    if (fabs(D) < 1e-6) {           // segment is parallel to plane
        if (N == 0)                      // segment lies in plane
            return 2;
        else
            return 0;                    // no intersection
    }
    // they are not parallel
    // compute intersect param
    float sI = N / D;
    if (sI < 0 || sI > 1)
        return 0;                        // no intersection

    *result = edgeFrom + sI * u;                  // compute segment intersect point
    return 1;
}

bool inCircle(vec2f_t center, float radius, vec2f_t point)
{
  vec2f_t delta = center - point;
  return delta.x() * delta.x() + delta.y() * delta.y() < radius * radius;
}

//Assumes conflict cylinder is placed at vertex. Check if edge intersects cylinder
bool ve_conflict(const ConflictCylinder& cyl, const vec3f_t& vertex, const vec3f_t& edgeFrom, const vec3f_t& edgeTo)
{
  // The cylinder is aligned with the z-axis, so our approach is as following:
  // 1. check if there is a line-segment/circle intersection
  // 2. If yes, check the z-coordinate for the interesection points
  // references:
  //  * https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
  //  * https://gamedev.stackexchange.com/questions/27210/segment-cylinder-intersection
  //  * https://stackoverflow.com/questions/4078401/trying-to-optimize-line-vs-cylinder-intersection


  // 1. check if there is a line-segment/circle intersection
  vec2f_t q(vertex.x(), vertex.y()); // center of circle
  vec2f_t p1(edgeFrom.x(), edgeFrom.y()); // start of line segment
  vec2f_t p2(edgeTo.x(), edgeTo.y()); // end of line segment
  vec2f_t v = p2 - p1;

  float a = v.dot(v);
  float b = 2 * v.dot(p1 - q);
  float c = p1.dot(p1) + q.dot(q) - 2 * p1.dot(q) - powf(cyl.radius, 2);
  float discriminant = powf(b, 2) - 4 * a * c;
  if (discriminant < 0) {
    return false;
  }

  float t1 = (-b + sqrtf(discriminant)) / (2 * a);
  float t2 = (-b - sqrtf(discriminant)) / (2 * a);

  float min_z_cylinder = vertex.z() - cyl.below;
  float max_z_cylinder = vertex.z() + cyl.above;

  // check candidate 1
  if (t1 >= 0 && t1 <= 1) {
    // compute potential intersection point
    vec3f_t pt = edgeFrom + (edgeTo - edgeFrom) * t1;
    // check z coordinate
    if (pt.z() >= min_z_cylinder && pt.z() <= max_z_cylinder) {
      return true;
    }
  }

  // check candidate 2
  if (t2 >= 0 && t2 <= 1) {
    // compute potential intersection point
    vec3f_t pt = edgeFrom + (edgeTo - edgeFrom) * t2;
    // check z coordinate
    if (pt.z() >= min_z_cylinder && pt.z() <= max_z_cylinder) {
      return true;
    }
  }

  // check top plane versus line segment
  vec3f resultPt;
  int res = intersect3D_SegmentPlane(edgeFrom, edgeTo, vec3f(vertex.x(), vertex.y(), max_z_cylinder), vec3f(0,0,1), &resultPt);
  if (res == 1
    && inCircle(q, cyl.radius, vec2f_t(resultPt.x(), resultPt.y()))) {
    return true;
  }

  // check bottom plane versus line segment
  res = intersect3D_SegmentPlane(edgeFrom, edgeTo, vec3f(vertex.x(), vertex.y(), min_z_cylinder), vec3f(0,0,1), &resultPt);
  if (res == 1
    && inCircle(q, cyl.radius, vec2f_t(resultPt.x(), resultPt.y()))) {
    return true;
  }

  // check if line segment is in plane
  if (vv_conflict(cyl, vertex, edgeFrom)
      || vv_conflict(cyl, vertex, edgeTo)) {
    return true;
  }

  return false;
}

size_t omp_ve_conflicts(const ConflictCylinder& cyl, const AgentType& typeA, const AgentType& typeB, AnnotatedRoadmap& g, size_t numJobs){

  //int nVerts = g.vertices.size();
  //int nEdges = g.edges.size();
  size_t nConflicts = 0;
  // SweptCylinderHull sweptHull;
  //vertex-edge
  #pragma omp parallel for num_threads(numJobs) reduction(+:nConflicts)
  for (size_t i = typeA.vStart; i < typeA.vEnd; ++i){ //agentA at vertex i
    for (size_t j = typeB.eStart; j < typeB.eEnd; ++j){ //agentB on edge j
      //check if cylinder at vertex 'i' conflicts with edge 'j'
      if(ve_conflict(cyl, g.vertices[i].pos, g.vertices[g.edges[j].from].pos, g.vertices[g.edges[j].to].pos)){
        g.vertices[i].edgeCollisions[0].insert(j);
        nConflicts += 1;
      }
    }
  }

  return nConflicts;
}

//Return false if no conflict, true if conflict
bool vv_conflict(const ConflictCylinder& cyl, const vec3f_t& cylPos, const vec3f_t& vPos){


  // check if point is it right height
  float top = cylPos.z() + cyl.above;
  float bottom = cylPos.z() - cyl.below;
  if (vPos.z() >= top || vPos.z() <= bottom) {
    return false;
  }

  // check if point is within radius:
  vec3f_t xyd(vPos.x() - cylPos.x(), vPos.y() - cylPos.y(), 0);
  float xyDist = xyd.norm();
  if (xyDist > cyl.radius){
    return false;
  }
  return true;
}

size_t omp_vv_conflicts(const ConflictCylinder& cyl, const AgentType& typeA, const AgentType& typeB, AnnotatedRoadmap& g, size_t numJobs){

  //int nVerts = g.vertices.size();
  size_t nConflicts = 0;
  //vertex-vertex
  #pragma omp parallel for num_threads(numJobs) reduction(+:nConflicts)
  for (size_t i = typeA.vStart; i < typeA.vEnd; ++i){ //agentA at vertex i
    for (size_t j = typeB.vStart; j < typeB.vEnd; ++j){ //agentB at vertex j
      if (i == j){
        continue;
      }
      //check if cylinder at vertex 'i' contains vertex 'j'
      if(vv_conflict(cyl, g.vertices[i].pos, g.vertices[j].pos)){
        g.vertices[i].vertexCollisions[0].insert(j);
        nConflicts += 1;
      }
    }
  }
  return nConflicts;
}

int main(int argc, char** argv) {

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string typesFile;
  std::string outputFile;
  std::string folder;
  size_t numJobs;


  desc.add_options()
      ("help", "produce help message")
      ("typesSet,t", po::value<std::string>(&typesFile)->required(), "input yaml for agent types, interaction definitions, roadmap assignments")
      ("output,o", po::value<std::string>(&outputFile)->required(), "output file for annotated graph")
      ("folder", po::value<std::string>(&folder)->required(), "folder where to find roadmap files")
      ("jobs,j", po::value<size_t>(&numJobs)->default_value(1), "number of jobs to run in parallel")
  ;

  try
  {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
  }
  catch(po::error& e)
  {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }


  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~ !!!DEBUG PAUSE!!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  // int debug = 0;
  // std::cout << "Debug: ";
  // std::cin >> debug;

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load Agent Types ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  std::vector<AgentType> agentTypes;
  std::unordered_map<std::string, int> type2id;
  YAML::Node yamlTypes = YAML::LoadFile(typesFile);
  std::cout << "Loading types ... " << std::endl;
  int nTypes = 0;
  for (const auto& node : yamlTypes["agentTypes"]) {

    std::string typeName = node["type"].as<std::string>();
    std::string subgraphPath = folder + "/" + node["roadmap"].as<std::string>();

    agentTypes.push_back(AgentType());
    agentTypes.back().type = typeName;
    agentTypes.back().subgraphYaml = subgraphPath;

    //interaction def for environment (unused; roadmaps do this implicitly now)
    // agentTypes.back().envSize.above = node["above"].as<float>();
    // agentTypes.back().envSize.below = node["below"].as<float>();
    // agentTypes.back().envSize.radius = node["radius"].as<float>();

    type2id.insert(std::make_pair(typeName,nTypes)); //used to parse interaction definitions

    nTypes++;
  }

  std::vector<std::vector< ConflictCylinder> > agentInteractions(nTypes, std::vector<ConflictCylinder>(nTypes));
  for (const auto& safety : yamlTypes["agentInteractions"]){

    std::string typeA = safety["typeA"].as<std::string>();
    std::string typeB = safety["typeB"].as<std::string>();

    //type check
    if (type2id.find(typeA) == type2id.end()){
      std::cerr << "Agent type \"" << typeA << "\" in interaction definition does not exist" << std::endl;
      return 1;
    }
    if (type2id.find(typeB) == type2id.end()){
      std::cerr << "Agent type \"" << typeB << "\" in interaction definition does not exist" << std::endl;
      return 1;
    }
    int typeAind = type2id[typeA];
    int typeBind = type2id[typeB];

    //add safe zone specification for each agent pair
    //Conflict cylinder definition cyl(above, below, radius)
    //  Cylinder defined for each agent type pair
    //  For types 'A' and 'B' the cylinder parameters are read as:
    //    Agent 'A' is required to be cyl.above z distance above 'B'
    //    Agent 'A' is required to be cyl.below z distance below 'B'
    //    Agent 'A' is required to be cyl.radius xy radius away from 'B'
    float above = safety["above"].as<float>();
    float below = safety["below"].as<float>();
    float radius = safety["radius"].as<float>();

    //Zone specification for agent 'A' interacting with agent 'B'
    agentInteractions[typeAind][typeBind] = ConflictCylinder(above,below,radius);

    if (typeAind != typeBind){
      //safe zones are symmetric
      // ex: if agent 'a' safeAbove for agent 'b' is 0.7, then agent 'b's safeBelow for agent 'a' is 0.7
      // this is kind of hacky way to implement this
      agentInteractions[typeBind][typeAind] = ConflictCylinder(below,above,radius);
    }
  }

  //maybe add a check to ensure all interactions are specified?

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~ Load Subgraphs, combine into larger graph ~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */


  AnnotatedRoadmap graph;

  for (auto& agent : agentTypes){
    //addSubgraph loads agentType graph, renames verts/edges using agent type name as a suffix
    //  also populates agentType subgraph range members
    graph.addSubgraph(agent);

    std::cout << agent.type << " vrange " << agent.vStart << ", " << agent.vEnd << std::endl;
    std::cout << agent.type << " erange " << agent.eStart << ", " << agent.eEnd << std::endl;
  }

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~ Container Setup For Conflict Checks ~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  // // this maps current agent types 'i' 'j' to the corresponding outer vector of collisions for that pair
  // //  there is one such index for every pair
  // std::vector<std::vector<int> > typePairToConflictIdx;
  // typePairToConflictIdx.resize(nTypes,std::vector<int>(nTypes));

  // //setup index map
  // int nTypePairs = 0;
  // for (int i = 0; i < nTypes; ++i){
  //   for (int j = i; j < nTypes; ++j){
  //     typePairId[i][j] = nTypePairs;
  //     typePairId[j][i] = nTypePairs; //symmetric relationship
  //     nTypePairs++;
  //   }
  // }

  //resize vertex/edge collision containers
  //   in subgraph version there is only 1 conflict set
  for (auto& vert : graph.vertices){
    vert.vertexCollisions.resize(1);
    vert.edgeCollisions.resize(1);
  }
  for (auto& edge : graph.edges){
    edge.vertexCollisions.resize(1);
    edge.edgeCollisions.resize(1);
  }

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~ Agent-Agent Conflicts ~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  //for every agent pair
  StopWatch timer;
  for (int a = 0; a < nTypes; a++){
    for (int b = a; b < nTypes; b++){
      //DEBUG
      // a = 1;
      // b = 1;
      std::cout << "Computing conflict pair: " << agentTypes[a].type + "-" + agentTypes[b].type << std::endl;

      //vertex-vertex
      size_t vvconflicts = omp_vv_conflicts(agentInteractions[a][b], agentTypes[a], agentTypes[b], graph, numJobs);

      //vertex-edge 1
      size_t veconflicts1 = omp_ve_conflicts(agentInteractions[a][b], agentTypes[a], agentTypes[b], graph, numJobs);

      //vertex-edge 2
      size_t veconflicts2 = omp_ve_conflicts(agentInteractions[b][a], agentTypes[b], agentTypes[a], graph, numJobs);

      //edge-edge
      size_t eeconflicts = omp_ee_conflicts(agentInteractions[a][b], agentTypes[a], agentTypes[b], graph, numJobs);


      //edge-edge
      //size_t eeconflicts = omp_ee_conflicts()

      //A vertex - B edge
      //size_t veconflicts = omp_ve_conflicts()

      //A edge - B vertex
      //veconflicts += omp_ee_conflicts()

      std::cout << "  Stats: " << std::endl;
      std::cout << "    VV: " << vvconflicts << std::endl;
      std::cout << "    VE1: " << veconflicts1 << std::endl;
      std::cout << "    VE2: " << veconflicts2 << std::endl;
      std::cout << "    EE: " << eeconflicts << std::endl;
      std::cout << "    Time: " << timer.seconds() << std::endl;
    }
  }
  std::cout << "Total Time: " << timer.seconds() << std::endl;


  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~ Agent-Environment Restrictions ~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  //now handled implicitly by subgraphs

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Save ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  graph.saveYaml(outputFile,agentTypes);


  return 0;
}
