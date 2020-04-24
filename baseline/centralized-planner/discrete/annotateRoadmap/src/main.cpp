#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <thread>

// Boost
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

// Eigen
#include <Eigen/Core>

// FCL Headers
#include <fcl/collision.h>
#include <fcl/collision_node.h>
#include <fcl/traversal/traversal_node_setup.h>
#include <fcl/continuous_collision.h>

//TODO fix this sloppy include
#include "../../generateRoadmap/src/fclHelper.h"

// Yaml
#include "yaml-cpp/yaml.h"

#include "libCommon/Timer.hpp"

using namespace std;
using namespace fcl;

typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> position_t;

Vec3f toVec3f(const position_t& pos)
{
  return Vec3f(pos.x(), pos.y(), pos.z());
}

struct agentType {
  agentType(
    const std::string& type,
    position_t conflictSize, //axis alligned ellipsoid
    position_t obstacleSize//axis alligned ellipsoid)
    ) : type(type), conflictSize(conflictSize), obstacleSize(obstacleSize)
  {}

  std::string type;
  position_t conflictSize;
  position_t obstacleSize;
};

struct vertex {
  vertex(
    const std::string& name,
    const position_t& pos)
    : name(name)
    , pos(pos)
  {
  }
  std::string name;
  position_t pos;
  std::vector<std::set<size_t> > edgeCollisions;
  std::vector<std::set<size_t> > vertexCollisions;
  std::set<std::string> typeRestrictions;
};

struct edge {
  edge(
    const std::string& name,
    size_t from,
    size_t to)
    : name(name)
    , from(from)
    , to(to)
  {
  }

  std::string name;
  size_t from;
  size_t to;
  std::vector<std::set<size_t> > edgeCollisions;
  std::vector<std::set<size_t> > vertexCollisions;
  std::set<std::string> typeRestrictions;
};

void vertexRestrictionChecker(
  size_t job,
  size_t numJobs,
  std::vector<vertex>& vertices,
  std::string& typeName,
  const CollisionGeometry* robot,
  const CollisionGeometry* environment)
{
  
  for (size_t i = job; i < vertices.size(); i += numJobs) {
    Transform3f robot_tf(toVec3f(vertices[i].pos));
    Transform3f environment_tf;
    CollisionRequest request;
    CollisionResult result;

    collide(environment,environment_tf,robot,robot_tf,request,result);

    if (result.isCollision()){
      vertices[i].typeRestrictions.insert(typeName);
    }
  }
}

void edgeRestrictionChecker(
  size_t job,
  size_t numJobs,
  std::vector<vertex>& vertices,
  std::vector<edge>& edges,
  std::string& typeName,
  const CollisionGeometry* robot,
  const CollisionGeometry* environment)
{
  
  for (size_t i = job; i < edges.size(); i += numJobs) {
    Transform3f robot_tf_beg(toVec3f(vertices[edges[i].from].pos));
    Transform3f robot_tf_end(toVec3f(vertices[edges[i].to].pos));

    Transform3f environment_tf;

    ContinuousCollisionRequest request;
    ContinuousCollisionResult result;

    continuousCollide(environment,environment_tf,environment_tf,robot,robot_tf_beg,robot_tf_end,request,result);

    bool collide = result.is_collide;
    if (collide){
      edges[i].typeRestrictions.insert(typeName);
    }
  }
}

void collisionCheckerVertexVertex(
  size_t job,
  size_t numJobs,
  size_t confIdx,
  std::vector<vertex>& vertices,
  std::vector<edge>& edges,
  const CollisionGeometry* robo1,
  const CollisionGeometry* robo2)
{
  for (size_t i = job; i < vertices.size(); i += numJobs) {
    for (size_t j = i + 1; j < vertices.size(); ++j) {
      fcl::Transform3f robot_tf1(toVec3f(vertices[i].pos));
      fcl::Transform3f robot_tf2(toVec3f(vertices[j].pos));
      CollisionRequest request;
      CollisionResult result;
      collide(robo1, robot_tf1, robo2, robot_tf2, request, result);

      if (result.isCollision()) {
        vertices[i].vertexCollisions[confIdx].insert(j);
      }

    }
  }

}

void collisionCheckerEdgeEdgeFCL(
  size_t job,
  size_t numJobs,
  size_t confIdx,
  std::vector<vertex>& vertices,
  std::vector<edge>& edges,
  const CollisionGeometry* robo1,
  const CollisionGeometry* robo2)
{
  // edge collision test
  // size_t numEdgeCollisions = 0;
  // Ellipsoid<double> robotd(rxy, rxy, rz);
  for (size_t i = job; i < edges.size(); i += numJobs) {
    for (size_t j = i + 1; j < edges.size(); ++j) {
      bool checkSameDirection = false;
      bool checkOppositeDirection = false;
      if (   edges[i].from == edges[j].from
          || edges[i].to == edges[j].to) {
        // check in opposite directions only
        checkOppositeDirection = true;
      } else if (   edges[i].to == edges[j].from
                 || edges[i].from == edges[j].to)
      {
        // check in same direction only
        checkSameDirection = true;
      } else {
        // check both directions
        checkOppositeDirection = true;
        checkSameDirection = true;
      }
      // std::cout << i << "," << j << " " << checkOppositeDirection << "," << checkSameDirection << std::endl;
      bool collide = false;

      if (checkSameDirection //) {
          && vertices[edges[i].from].vertexCollisions[confIdx].find(edges[j].from) == vertices[edges[i].from].vertexCollisions[confIdx].end()
          && vertices[edges[i].to].vertexCollisions[confIdx].find(edges[j].to) == vertices[edges[i].to].vertexCollisions[confIdx].end()) {

        Transform3f robot_tf1_beg(toVec3f(vertices[edges[i].from].pos));
        Transform3f robot_tf1_end(toVec3f(vertices[edges[i].to].pos));
        Transform3f robot_tf2_beg(toVec3f(vertices[edges[j].from].pos));
        Transform3f robot_tf2_end(toVec3f(vertices[edges[j].to].pos));

        ContinuousCollisionRequest request;
        ContinuousCollisionResult result;
        continuousCollide(robo1, robot_tf1_beg, robot_tf1_end,
                          robo2, robot_tf2_beg, robot_tf2_end,
                          request, result);

        collide |= result.is_collide;
        // std::cout << "  s: " << result.is_collide << std::endl;
      }

      if (!collide && checkOppositeDirection // ) {
          && vertices[edges[i].from].vertexCollisions[confIdx].find(edges[j].to) == vertices[edges[i].from].vertexCollisions[confIdx].end()
          && vertices[edges[i].to].vertexCollisions[confIdx].find(edges[j].from) == vertices[edges[i].to].vertexCollisions[confIdx].end()) {

        Transform3f robot_tf1_beg(toVec3f(vertices[edges[i].to].pos));
        Transform3f robot_tf1_end(toVec3f(vertices[edges[i].from].pos));
        Transform3f robot_tf2_beg(toVec3f(vertices[edges[j].from].pos));
        Transform3f robot_tf2_end(toVec3f(vertices[edges[j].to].pos));

        ContinuousCollisionRequest request;
        ContinuousCollisionResult result;
        continuousCollide(robo1, robot_tf1_beg, robot_tf1_end,
                          robo2, robot_tf2_beg, robot_tf2_end,
                          request, result);

        collide |= result.is_collide;
        // std::cout << "  o: " << result.is_collide << std::endl;
      }

      if (collide) {
        edges[i].edgeCollisions[confIdx].insert(j);
        // edges[j].collisions.insert(i);
        // ++numEdgeCollisions;
      }
    }
  }
}


void collisionCheckerEdgeVertexFCL(
  size_t job,
  size_t numJobs,
  size_t confIdx,
  std::vector<vertex>& vertices,
  std::vector<edge>& edges,
  const CollisionGeometry* robo1,
  const CollisionGeometry* robo2)
{
  // edge collision test
  for (size_t i = job; i < edges.size(); i += numJobs) {
    for (size_t j = 0; j < vertices.size(); ++j) {
      bool collide = false;

      if ( edges[i].from != j 
        && edges[i].to != j) {

        Transform3f robot_tf1_beg(toVec3f(vertices[edges[i].from].pos));
        Transform3f robot_tf1_end(toVec3f(vertices[edges[i].to].pos));
        Transform3f robot_tf2_beg(toVec3f(vertices[j].pos));
        Transform3f robot_tf2_end(toVec3f(vertices[j].pos));

        ContinuousCollisionRequest request;
        ContinuousCollisionResult result;
        continuousCollide(robo1, robot_tf1_beg, robot_tf1_end,
                          robo2, robot_tf2_beg, robot_tf2_end,
                          request, result);

        collide |= result.is_collide;
      }

      if (collide) {
        edges[i].vertexCollisions[confIdx].insert(j);
      }
    }
  }
}

int main(int argc, char** argv) {


  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string graphFile;
  std::string enviroFile;
  std::string typesFile;
  std::string outputFile;
  float ellipsoidScale;
  
  size_t numJobs;
  bool sweptEllipsoid;
  desc.add_options()
      ("help", "produce help message")
      ("graph,g", po::value<std::string>(&graphFile)->required(), "input file for graph")
      ("environment,e", po::value<std::string>(&enviroFile)->required(), "environment file for restriction calculation")
      ("typesSet,t", po::value<std::string>(&typesFile)->required(), "input file for agent ellipsoid sizes")
      ("output,o", po::value<std::string>(&outputFile)->required(), "output file for annotated graph")
      ("jobs,j", po::value<size_t>(&numJobs)->default_value(1), "number of jobs to run in parallel")
      ("ellipsoidScale",po::value<float>(&ellipsoidScale)->default_value(1.0), "scale ellipsoids when performing checks (does not affect output file)")
      ("sweptEllipsoid", po::bool_switch(&sweptEllipsoid)->default_value(false), "use sweptEllipsoid rather than FCL for edge collision checking")
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

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~ load graph ~~~~~~~~~~~~~~~~~~~~~~~~~ */
  Timer totalTimer;

  std::vector<vertex> vertices;
  std::vector<edge> edges;
  YAML::Node yamlGraph = YAML::LoadFile(graphFile);
  std::unordered_map<std::string, size_t> vNameToIdx;

  std::cout << "Loading vertices ... " << std::endl;

  for (const auto& node : yamlGraph["vertices"]) {
    const auto& pos = node["pos"];
    std::string name = node["name"].as<std::string>();
    position_t p(
      pos[0].as<float>(),
      pos[1].as<float>(),
      pos[2].as<float>());
    vertices.push_back(vertex(name, p));
    vNameToIdx[name] = vertices.size() - 1;
  }

  std::cout << "Loading edges ... " << std::endl;

  for (const auto& node : yamlGraph["edges"]) {
    std::string name = node["name"].as<std::string>();
    std::string fromName = node["from"].as<std::string>();
    std::string toName = node["to"].as<std::string>();
    size_t from = vNameToIdx[fromName];
    size_t to = vNameToIdx[toName];
    if (    from >= vertices.size()
         || to >= vertices.size()
         || from == to) {
      std::cerr << "invalid edge! " << node << std::endl;
      continue;
    }
    edges.push_back(edge(name, from, to));
  }

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~ load agent types ~~~~~~~~~~~~~~~~~~~~~~~~~ */

  std::vector<agentType> agentTypes;
  YAML::Node yamlAgents = YAML::LoadFile(typesFile);

  std::cout << "Loading types ... " << std::endl;

  std::unordered_map<std::string, size_t> emptyStats; //only used for stats
  for (const auto& node : yamlAgents["agentTypes"]) {
    std::string type = node["type"].as<std::string>();

    const auto& csz = node["conflictSize"];
    position_t confSize(csz[0].as<float>(),csz[1].as<float>(),csz[2].as<float>());
      
    const auto& osz = node["obstacleSize"];
    position_t obsSize(osz[0].as<float>(),osz[1].as<float>(),osz[2].as<float>()); 

    agentTypes.push_back(agentType(type,confSize,obsSize));
    emptyStats.insert(std::make_pair(type,0));
  }


  /* ~~~~~~~~~~~~~~~~~~~~~~~~~ Vertex Restrictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  //Calculate vertex restrictions for agent types
  //  A type is restricted to visit a particular vertex if it collides with the environment

  /* ~~~~ Open / Load Environment File ~~~~ */
  std::cout << "Loading Environment ... " << std::endl;
  CollisionGeometry* environment = fclHelper::createCollisionGeometryFromFile(enviroFile);

  std::cout << "Calculating type restrictions ... " << std::endl;
  Timer vertRestrictTimer;
  
  //iterate over agent types
  for (size_t i = 0; i < agentTypes.size(); ++i){
    //Ellipsoid robot(agentTypes[i].rx,agentTypes[i].ry,agentTypes[i].rz);
    Ellipsoid robot(agentTypes[i].obstacleSize.x() * ellipsoidScale,agentTypes[i].obstacleSize.y() * ellipsoidScale,agentTypes[i].obstacleSize.z() * ellipsoidScale);

    if (numJobs == 1){
      vertexRestrictionChecker(0,1,vertices,agentTypes[i].type,&robot,environment);
    }
    else{
      std::vector<std::thread> threads;
      for (size_t j = 0; j < numJobs; ++j) {
          threads.push_back(std::thread(vertexRestrictionChecker, j, numJobs, std::ref(vertices), std::ref(agentTypes[i].type), &robot, environment));
      }
      // wait for all threads
      for (auto& thread : threads) {
        thread.join();
      }
      threads.clear();
    }
  }

  /* ~~~~~ Stats ~~~~~ */
  vertRestrictTimer.stop();
  std::cout << std::setprecision(4);
  //count vertex restrictions
  std::unordered_map<std::string, size_t> vertRestrictStats(emptyStats);
  for (auto& vert : vertices){
    for (auto& restrictedType : vert.typeRestrictions){
      vertRestrictStats[restrictedType] += 1;
    }
  }
  std::cout << "Vertex Restriction Stats: " << std::endl;
  std::cout << "   Time: " << vertRestrictTimer.elapsedSeconds() << " s" << std::endl;
  for (auto& stat : vertRestrictStats){
    std::cout << "   " << stat.first << " : " << stat.second << std::endl;
  }

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~ Edge Restrictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  //Calculate edge restrictions for agent types
  //  A type is restricted to visit a particular edge if it collides with the environment during traversal

  Timer edgeRestrictTimer;
  //iterate over agent types
  for (size_t i = 0; i < agentTypes.size(); ++i){
    //Ellipsoid robot(agentTypes[i].rx,agentTypes[i].ry,agentTypes[i].rz);
    Ellipsoid robot(agentTypes[i].obstacleSize.x() * ellipsoidScale,agentTypes[i].obstacleSize.y() * ellipsoidScale,agentTypes[i].obstacleSize.z() * ellipsoidScale);
    if (numJobs == 1){
      edgeRestrictionChecker(0,1,vertices,edges,agentTypes[i].type,&robot,environment);
    }
    else{
      std::vector<std::thread> threads;
      for (size_t j = 0; j < numJobs; ++j) {
          threads.push_back(std::thread(edgeRestrictionChecker, j, numJobs, std::ref(vertices), std::ref(edges), std::ref(agentTypes[i].type), &robot, environment));
      }
      // wait for all threads
      for (auto& thread : threads) {
        thread.join();
      }
      threads.clear();
    }
  }

  /* ~~~~~ Stats ~~~~~ */
  edgeRestrictTimer.stop();
  std::cout << std::setprecision(4);
  //count vertex restrictions
  std::unordered_map<std::string, size_t> edgeRestrictStats(emptyStats);
  for (auto& edge : edges){
    for (auto& restrictedType : edge.typeRestrictions){
      edgeRestrictStats[restrictedType] += 1;
    }
  }
  std::cout << "Edge Restriction Stats: " << std::endl;
  std::cout << "   Time: " << edgeRestrictTimer.elapsedSeconds() << " s" << std::endl;
  for (auto& stat : edgeRestrictStats){
    std::cout << "   " << stat.first << " : " << stat.second << std::endl;
  }

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~ Conflict Checks ~~~~~~~~~~~~~~~~~~~~~~~~~ */

  //this maps current agent types 'i' 'j' to the corresponding outer vector of collisions for that pair
  //  there is one such index for every pair
  std::vector<std::vector<size_t>> typePairToConflictIdx;
  typePairToConflictIdx.resize(agentTypes.size(),std::vector<size_t>(agentTypes.size()));

  //setup index map
  size_t currConfIdx = 0;
  for (size_t i = 0; i < agentTypes.size(); ++i){
    for (size_t j = i; j < agentTypes.size(); ++j){
      typePairToConflictIdx[i][j] = currConfIdx;
      typePairToConflictIdx[j][i] = currConfIdx; //symmetric relationship
      currConfIdx++;
    }
  }
  size_t numPairs = currConfIdx; //for clarity

  //resize vertex/edge collision containers
  for (auto& vert : vertices){
    vert.vertexCollisions.resize(numPairs);
    vert.edgeCollisions.resize(numPairs);
  }
  for (auto& edge : edges){
    edge.vertexCollisions.resize(numPairs);
    edge.edgeCollisions.resize(numPairs);
  }

  std::cout << "Calculating conflicts ... " << std::endl;

  //Iterate over agentType pairs
  for (size_t i = 0; i < agentTypes.size(); ++i){
    for (size_t j = i; j < agentTypes.size(); ++j){
      
      size_t confIdx = typePairToConflictIdx[i][j];

      //Robot ellipsoids
      Ellipsoid robo1(agentTypes[i].conflictSize.x() * ellipsoidScale,agentTypes[i].conflictSize.y() * ellipsoidScale,agentTypes[i].conflictSize.z() * ellipsoidScale);
      Ellipsoid robo2(agentTypes[j].conflictSize.x() * ellipsoidScale,agentTypes[j].conflictSize.y() * ellipsoidScale,agentTypes[j].conflictSize.z() * ellipsoidScale);

      /* ~~~~~~~~~~~~~~~~~~~~~~~~~ Checks ~~~~~~~~~~~~~~~~~~~~~~~~~ */

      Timer pairTimer;

      if (numJobs == 1) {
        // Find all vertex/vertex collisions
        collisionCheckerVertexVertex(0, 1, confIdx, vertices, edges, &robo1, &robo2);

        // merge results
        for (size_t i = 0; i < vertices.size(); ++i) {
          for (size_t c : vertices[i].vertexCollisions[confIdx]) {
            vertices[c].vertexCollisions[confIdx].insert(i);
          }
        }

        // Find all edge/edge collisions (requires to know all vertex collisions)
        if (sweptEllipsoid) {
          //collisionCheckerEdgeEdgeSweptEllipsoid(0, 1, confIdx, vertices, edges, rxy, rz);
        } else {
          collisionCheckerEdgeEdgeFCL(0, 1, confIdx, vertices, edges, &robo1, &robo2);
        }

        // merge results
        for (size_t i = 0; i < edges.size(); ++i) {
          for (size_t c : edges[i].edgeCollisions[confIdx]) {
            edges[c].edgeCollisions[confIdx].insert(i);
          }
        }

        // Find all edge/vertex collisions
        if (sweptEllipsoid) {
          //collisionCheckerEdgeVertexSweptEllipsoid(0, 1, confIdx, vertices, edges, rxy, rz);
        } else {
          collisionCheckerEdgeVertexFCL(0, 1, confIdx, vertices, edges, &robo1, &robo2);
        }

        // merge results
        for (size_t i = 0; i < edges.size(); ++i) {
          for (size_t c : edges[i].vertexCollisions[confIdx]) {
            vertices[c].edgeCollisions[confIdx].insert(i);
          }
        }

      } 
      else {
        // Find all vertex/vertex collisions
        std::vector<std::thread> threads;
        for (size_t i = 0; i < numJobs; ++i) {
          threads.push_back(std::thread(collisionCheckerVertexVertex, i, numJobs, confIdx, std::ref(vertices), std::ref(edges), &robo1, &robo2));
        }
        // wait for all threads
        for (auto& thread : threads) {
          thread.join();
        }
        threads.clear();

        // merge results
        for (size_t i = 0; i < vertices.size(); ++i) {
          for (size_t c : vertices[i].vertexCollisions[confIdx]) {
            vertices[c].vertexCollisions[confIdx].insert(i);
          }
        }

        // Find all edge/edge collisions (requires to know all vertex collisions)
        for (size_t i = 0; i < numJobs; ++i) {
          if (sweptEllipsoid) {
            //threads.push_back(std::thread(collisionCheckerEdgeEdgeSweptEllipsoid, i, confIdx, numJobs, std::ref(vertices), std::ref(edges), rxy, rz));
          } else {
            threads.push_back(std::thread(collisionCheckerEdgeEdgeFCL, i, numJobs, confIdx, std::ref(vertices), std::ref(edges), &robo1, &robo2));
          }
        }
        // wait for all threads
        for (auto& thread : threads) {
          thread.join();
        }
        threads.clear();

        // merge results
        for (size_t i = 0; i < edges.size(); ++i) {
          for (size_t c : edges[i].edgeCollisions[confIdx]) {
            edges[c].edgeCollisions[confIdx].insert(i);
          }
        }

        // Find all edge/vertex collisions
        for (size_t i = 0; i < numJobs; ++i) {
          if (sweptEllipsoid) {
            //threads.push_back(std::thread(collisionCheckerEdgeVertexSweptEllipsoid, i, confIdx, numJobs, std::ref(vertices), std::ref(edges), rxy, rz));
          } else {
            threads.push_back(std::thread(collisionCheckerEdgeVertexFCL, i, numJobs, confIdx, std::ref(vertices), std::ref(edges), &robo1, &robo2));
          }
        }
        // wait for all threads
        for (auto& thread : threads) {
          thread.join();
        }
        threads.clear();

        // merge results
        for (size_t i = 0; i < edges.size(); ++i) {
          for (size_t c : edges[i].vertexCollisions[confIdx]) {
            vertices[c].edgeCollisions[confIdx].insert(i);
          }
        }
      }

      pairTimer.stop();

      /* ~~~~~~~~~~~~~~~~~~~~~~~~~ Stats ~~~~~~~~~~~~~~~~~~~~~~~~~ */

      size_t vertexVertexConflicts = 0;
      for (size_t i = 0; i < vertices.size(); ++i) {
        vertexVertexConflicts += vertices[i].vertexCollisions[confIdx].size();
      }
      size_t edgeEdgeConflicts = 0;
      for (size_t i = 0; i < edges.size(); ++i) {
        edgeEdgeConflicts += edges[i].edgeCollisions[confIdx].size();
      }
      size_t edgeVertexConflicts = 0;
      for (size_t i = 0; i < edges.size(); ++i) {
        edgeVertexConflicts += edges[i].vertexCollisions[confIdx].size();
      }

      std::string pairStr = agentTypes[i].type + "-" + agentTypes[j].type;
      std::cout << std::setprecision(4);
      std::cout << pairStr << " Conflict Stats: " << std::endl;
      std::cout << "   Time: " << pairTimer.elapsedSeconds() << " s" << std::endl;
      std::cout << "   V-V Conflicts: " << vertexVertexConflicts << "  ~  Avg Conf/Vert : " << vertexVertexConflicts/((float)vertices.size()) << std::endl;
      std::cout << "   E-E Conflicts: " << edgeEdgeConflicts << "  ~  Avg Conf/Edge : " << edgeEdgeConflicts/((float)edges.size()) << std::endl;
      std::cout << "   E-V Conflicts: " << edgeVertexConflicts << "  ~  Avg Conf/Edge : " << edgeVertexConflicts/((float)edges.size()) << std::endl;
      std::cout << "   E-V Conflicts: " << edgeVertexConflicts << "  ~  Avg Conf/Vert : " << edgeVertexConflicts/((float)vertices.size()) << std::endl;
    }
  }

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~ Write Annotated Graph File ~~~~~~~~~~~~~~~~~~~~~~~~~ */
  YAML::Emitter out;
  out.SetIndent(2);

  out << YAML::BeginMap;//Begin file

  //Output agent types in this annotated graph file
  out << YAML::Key << "agentTypes";
  out << YAML::Value << YAML::BeginSeq; //begin types sequence
  for (auto& ag : agentTypes){
    out << YAML::BeginMap; //begin type map
    out << YAML::Key << "type" << YAML::Value << ag.type;
    out << YAML::Key << "conflictSize";
    out << YAML::Value << YAML::Flow << YAML::BeginSeq << ag.conflictSize.x() << ag.conflictSize.y() << ag.conflictSize.z() << YAML::EndSeq;
    out << YAML::Key << "obstacleSize";
    out << YAML::Value << YAML::Flow << YAML::BeginSeq << ag.obstacleSize.x() << ag.obstacleSize.y() << ag.obstacleSize.z() << YAML::EndSeq;
    out << YAML::EndMap; //end type map
  }
  out << YAML::EndSeq; //end agentTypes sequence
  

  //Output roadmap with conflicts and restrictions
  out << YAML::Key << "annotatedRoadmap";
  out << YAML::BeginMap; //begin annotation map

  out << YAML::Key << "vertices";
  out << YAML::Value << YAML::BeginSeq; //begin vertex annotations
  for (size_t i = 0; i < vertices.size(); ++i) {
    out << YAML::BeginMap; //begin vertex map

    out << YAML::Key << "name" << YAML::Value << vertices[i].name;
    out << YAML::Key << "pos" << YAML::Value << YAML::Flow << YAML::BeginSeq << vertices[i].pos.x() << vertices[i].pos.y() << vertices[i].pos.z() << YAML::EndSeq;

    out << YAML::Key << "typeRestrictions";
    out << YAML::Value << YAML::Flow << YAML::BeginSeq; // begin type restrictions
    for (auto& typ : vertices[i].typeRestrictions){
      out << typ;
    }
    out << YAML::EndSeq; //end type restrictions

    //for each agent pair, output vertex and edge conflicts for this vertex
    out << YAML::Key << "conflicts";
    out << YAML::Value << YAML::BeginSeq; //begin type pair sequence
    for (size_t j = 0; j < agentTypes.size(); ++j){
      for (size_t k = j; k < agentTypes.size(); ++k){
        size_t idx = typePairToConflictIdx[j][k];
        out << YAML::BeginMap; //begin conflict map for this pair
        out << YAML::Key << "typePair" << YAML::Value << YAML::Flow << YAML::BeginSeq << agentTypes[j].type << agentTypes[k].type << YAML::EndSeq;

        out << YAML::Key << "vertexConflicts";
        out << YAML::Value << YAML::Flow << YAML::BeginSeq; //begin v-v conflicts
        for (size_t c : vertices[i].vertexCollisions[idx]){
          out << vertices[c].name;
        }
        out << YAML::EndSeq; //end v-v conflicts

        out << YAML::Key << "edgeConflicts";
        out << YAML::Value << YAML::Flow << YAML::BeginSeq; //begin v-e conflicts
        for (size_t c : vertices[i].edgeCollisions[idx]){
          out << edges[c].name;
        }
        out << YAML::EndSeq; //end v-e conflicts

        out << YAML::EndMap; //end conflict map for this pair
      }
    }
    out << YAML::EndSeq; //end pair sequence

    out << YAML::EndMap; //end vertex map

  }
  out << YAML::EndSeq; // end vertex sequence
  
  out << YAML::Key << "edges";
  out << YAML::Value << YAML::BeginSeq; //begin edge annotations
  for (size_t i = 0; i < edges.size(); ++i) {
    out << YAML::BeginMap; //begin edge map

    out << YAML::Key << "name" << YAML::Value << edges[i].name;
    out << YAML::Key << "from" << YAML::Value << vertices[edges[i].from].name;
    out << YAML::Key << "to" << YAML::Value << vertices[edges[i].to].name;

    out << YAML::Key << "typeRestrictions";
    out << YAML::Value << YAML::Flow << YAML::BeginSeq; // begin type restrictions
    for (auto& typ : edges[i].typeRestrictions){
      out << typ;
    }
    out << YAML::EndSeq; //end type restrictions

    //for each agent pair, output vertex and edge conflicts for this vertex
    out << YAML::Key << "conflicts";
    out << YAML::Value << YAML::BeginSeq; //begin type pair sequence
    for (size_t j = 0; j < agentTypes.size(); ++j){
      for (size_t k = j; k < agentTypes.size(); ++k){
        size_t idx = typePairToConflictIdx[j][k];
        out << YAML::BeginMap; //begin conflict map for this pair
        out << YAML::Key << "typePair" << YAML::Value << YAML::Flow << YAML::BeginSeq << agentTypes[j].type << agentTypes[k].type << YAML::EndSeq;

        out << YAML::Key << "vertexConflicts";
        out << YAML::Value << YAML::Flow << YAML::BeginSeq; //begin e-v conflicts
        for (size_t c : edges[i].vertexCollisions[idx]){
          out << vertices[c].name;
        }
        out << YAML::EndSeq; //end v-v conflicts

        out << YAML::Key << "edgeConflicts";
        out << YAML::Value << YAML::Flow << YAML::BeginSeq; //begin e-e conflicts
        for (size_t c : edges[i].edgeCollisions[idx]){
          out << edges[c].name;
        }
        out << YAML::EndSeq; //end v-e conflicts

        out << YAML::EndMap; //end conflict map for this pair
      }
    }
    out << YAML::EndSeq; //end pair sequence

    out << YAML::EndMap; //end edge map

  }
  out << YAML::EndSeq; //end edge sequence

  out << YAML::EndMap; //end annotation map

  out << YAML::EndMap; //end file

  //Save output
  ofstream ofstr(outputFile.c_str());
  ofstr << out.c_str();

  totalTimer.stop();
  std::cout << "Total Elapsed Time: " << totalTimer.elapsedSeconds() << std::endl;

  return 0;
}