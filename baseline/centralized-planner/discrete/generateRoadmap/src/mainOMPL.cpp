#include <iostream>
#include <unordered_map>
#include <unordered_set>

// Boost
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

// Eigen
#include <Eigen/Core>

// FCL Headers
#include <fcl/collision.h>
#include <fcl/collision_node.h>
#include <fcl/traversal/traversal_node_setup.h>
#include <fcl/continuous_collision.h>

// Assimp
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

// Yaml
#include <yaml-cpp/yaml.h>

// Nanoflann
#include <nanoflann.hpp>
#include "KDTreeVectorOfVectorsAdaptor.h"

// Octomap
#include <octomap/octomap.h>
#include <octomap/OcTree.h>

// OMPL
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/prm/SPARS.h>
#include <ompl/geometric/planners/prm/SPARStwo.h>



using namespace std;
using namespace fcl;
using namespace nanoflann;
using namespace ompl;

typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> position_t;

void extractVerticesAndTrianglesRecursive(
  const aiScene *scene,
  const aiNode *node,
  std::vector<Vec3f>& vertices,
  std::vector<Triangle>& triangles)
{
  for (size_t i = 0 ; i < node->mNumMeshes; ++i)
  {
    const aiMesh* a = scene->mMeshes[node->mMeshes[i]];
    size_t offset = vertices.size();

    for (size_t j = 0; j < a->mNumVertices; ++j) {
      vertices.push_back(
        Vec3f(
          a->mVertices[j].x,
          a->mVertices[j].y,
          a->mVertices[j].z));
    }

    for (size_t j = 0 ; j < a->mNumFaces ; ++j) {
      if (a->mFaces[j].mNumIndices == 3) {
        triangles.push_back(Triangle(
          a->mFaces[j].mIndices[0] + offset,
          a->mFaces[j].mIndices[1] + offset,
          a->mFaces[j].mIndices[2] + offset));
      }
    }
  }

  for (unsigned int n = 0; n < node->mNumChildren; ++n) {
    extractVerticesAndTrianglesRecursive(scene, node->mChildren[n], vertices, triangles);
  }
}

CollisionGeometry* createModelFromMeshFile(const std::string& fileName)
{
  std::string extension = boost::filesystem::extension(fileName);

  if (extension == ".bt") {
    const auto model = new OcTree(std::shared_ptr<const octomap::OcTree>(new octomap::OcTree(fileName)));
    return model;
  } else {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(fileName.c_str(),
      aiProcess_Triangulate            |
      aiProcess_JoinIdenticalVertices  |
      aiProcess_SortByPType            |
      aiProcess_OptimizeGraph          |
      aiProcess_OptimizeMeshes);

    std::vector<Vec3f> vertices;
    std::vector<Triangle> triangles;

    // code to set the vertices and triangles
    if (scene && scene->mRootNode) {
      extractVerticesAndTrianglesRecursive(scene, scene->mRootNode, vertices, triangles);

      // add the mesh data into the BVHModel structure
      const auto model = new BVHModel<OBBRSS>();
      model->beginModel();
      model->addSubModel(vertices, triangles);
      model->endModel();
      model->computeLocalAABB();

      return model;
    }
  }

  return nullptr;

  // std::cout << "Vertices" << std::endl;
  // size_t i = 0;
  // for (const auto& v : vertices) {
  //   std::cout << i << " " << v[0] << "," << v[1] << "," << v[2] << std::endl;

  //   ++i;
  // }
  // std::cout << "Triangles" << std::endl;
  // i = 0;
  // for (const auto& t : triangles) {
  //   std::cout << i << " " << t[0] << "," << t[1] << "," << t[2] << std::endl;
  //   ++i;
  // }


  // BVHModel is a template class for mesh geometry, for default OBBRSS template is used


}

struct range {
  double min;
  double max;
};

void boundingBox(const BVHModel<OBBRSS>& model, std::array<range, 3>& result)
{
  for (auto& r : result) {
    r.min = std::numeric_limits<float>::max();
    r.max = std::numeric_limits<float>::min();
  }

  for (int i = 0; i < model.num_vertices; ++i) {
    const Vec3f& v = model.vertices[i];
    for (size_t d = 0; d < 3; ++d) {
      result[d].min = std::min(result[d].min, v[d]);
      result[d].max = std::max(result[d].max, v[d]);
    }
  }
}

// see http://motion.me.ucsb.edu/book-lrpk/pdfs/LecturesPlanningKinematics-FB+SLS-v0.91.pdf, p91
// computes i'th halton number for given prime base p
double halton(int i, int p)
{
  double S = 0;
  int i_tmp = i;
  double f = 1.0 / p;
  while(i_tmp > 0) {
    std::div_t result = std::div(i_tmp, p);
    int q = result.quot;
    int r = result.rem;
    S += f * r;
    i_tmp = q;
    f /= p;
  }
  return S;
}

Vec3f toVec3f(const position_t& pos)
{
  return Vec3f(pos.x(), pos.y(), pos.z());
}

std::string getVertexName(
  size_t i,
  const std::vector<std::string>& vertexNames)
{
  if (i < vertexNames.size()) {
    return vertexNames[i];
  } else {
    return "v" + std::to_string(i);
  }
}

class MyStateValidityChecker
  : public ompl::base::StateValidityChecker
{
public:
  MyStateValidityChecker(
    base::SpaceInformationPtr si,
    CollisionGeometry* environment,
    CollisionGeometry* robot)
    : StateValidityChecker(si)
    , m_environment(environment)
    , m_robot(robot)
  {
  }

  bool isValid (const ompl::base::State* state) const
  {
    if (m_environment && m_robot) {
      const ompl::base::RealVectorStateSpace::StateType* typedState = state->as<ompl::base::RealVectorStateSpace::StateType>();

      fcl::Transform3f env_tf;
      fcl::Transform3f robot_tf(Vec3f((*typedState)[0], (*typedState)[1], (*typedState)[2]));
      CollisionRequest request;
      CollisionResult result;
      collide(m_environment, env_tf, m_robot, robot_tf, request, result);
      return !result.isCollision();
    }
    return true;
  }

private:
  CollisionGeometry* m_environment;
  CollisionGeometry* m_robot;
};

class MyMotionValidator
  : public ompl::base::MotionValidator
{
public:
  MyMotionValidator(
    base::SpaceInformationPtr si,
    CollisionGeometry* environment,
    CollisionGeometry* robot)
    : MotionValidator(si)
    , m_environment(environment)
    , m_robot(robot)
  {
  }

  bool checkMotion(const ompl::base::State* s1, const ompl::base::State* s2) const
  {
    if (m_environment && m_robot) {
      const ompl::base::RealVectorStateSpace::StateType* typedS1 = s1->as<ompl::base::RealVectorStateSpace::StateType>();
      const ompl::base::RealVectorStateSpace::StateType* typedS2 = s2->as<ompl::base::RealVectorStateSpace::StateType>();

      Transform3f env_tf_beg;
      Transform3f env_tf_end;
      Transform3f robot_tf_beg(Vec3f((*typedS1)[0], (*typedS1)[1], (*typedS1)[2]));
      Transform3f robot_tf_end(Vec3f((*typedS2)[0], (*typedS2)[1], (*typedS2)[2]));

      ContinuousCollisionRequest request;
      ContinuousCollisionResult result;
      continuousCollide(m_environment, env_tf_beg, env_tf_end,
                        m_robot, robot_tf_beg, robot_tf_end,
                        request, result);
      return !result.is_collide;
    }
    return true;
  }

  bool checkMotion(const ompl::base::State* s1, const ompl::base::State* s2, std::pair<ompl::base::State*, double>& lastValid) const
  {
    if (m_environment && m_robot) {
      const ompl::base::RealVectorStateSpace::StateType* typedS1 = s1->as<ompl::base::RealVectorStateSpace::StateType>();
      const ompl::base::RealVectorStateSpace::StateType* typedS2 = s2->as<ompl::base::RealVectorStateSpace::StateType>();

      Transform3f env_tf_beg;
      Transform3f env_tf_end;
      Transform3f robot_tf_beg(Vec3f((*typedS1)[0], (*typedS1)[1], (*typedS1)[2]));
      Transform3f robot_tf_end(Vec3f((*typedS2)[0], (*typedS2)[1], (*typedS2)[2]));

      ContinuousCollisionRequest request;
      ContinuousCollisionResult result;
      continuousCollide(m_environment, env_tf_beg, env_tf_end,
                        m_robot, robot_tf_beg, robot_tf_end,
                        request, result);
      // base::ScopedState<base::RealVectorStateSpace> contactPoint(space);
      ompl::base::State* contactPoint = si_->allocState();
      ompl::base::RealVectorStateSpace::StateType* contactPointTyped = contactPoint->as<ompl::base::RealVectorStateSpace::StateType>();
      for (size_t i = 0; i < 3; ++i) {
        (*contactPointTyped)[i] = result.contact_tf2.getTranslation()[i];
      }
      lastValid.first = contactPoint;
      lastValid.second = result.time_of_contact;
      return !result.is_collide;
    }
    return true;
  }

private:
  CollisionGeometry* m_environment;
  CollisionGeometry* m_robot;
};


int main(int argc, char** argv) {


  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string environmentFile;
  std::string robotFile;
  std::string addVerticesFile;
  std::string outputFile;
  size_t numPoints;
  double searchRadius;
  double xmin,xmax,ymin,ymax,zmin,zmax;
  desc.add_options()
      ("help", "produce help message")
      ("environment,e", po::value<std::string>(&environmentFile)->required(), "input file for environment (STL)")
      ("robot,r", po::value<std::string>(&robotFile)->required(), "input file for robot (STL)")
      ("addVertices,a", po::value<std::string>(&addVerticesFile), "input file for additional vertices (YAML)")
      ("output,o", po::value<std::string>(&outputFile)->required(), "output file for graph (YAML)")
      ("numPoints", po::value<size_t>(&numPoints)->default_value(100), "Number of points to sample in bounding box")
      ("searchRadius", po::value<double>(&searchRadius)->default_value(0.5), "kd tree search radius")
      ("xmin", po::value<double>(&xmin)->default_value(0.0), "bounding box minimum x")
      ("xmax", po::value<double>(&xmax)->default_value(1.0), "bounding box maximum x")
      ("ymin", po::value<double>(&ymin)->default_value(0.0), "bounding box minimum y")
      ("ymax", po::value<double>(&ymax)->default_value(1.0), "bounding box maximum y")
      ("zmin", po::value<double>(&zmin)->default_value(0.0), "bounding box minimum z")
      ("zmax", po::value<double>(&zmax)->default_value(1.0), "bounding box maximum z")
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

  CollisionGeometry* environment = createModelFromMeshFile(environmentFile);

  // return 0;
  CollisionGeometry* robot = createModelFromMeshFile(robotFile);

  // Sphere* robot = new Sphere(0.15);

  std::array<range, 3> bbox;
  #if 0
  boundingBox(*environment, bbox);
  #else
  bbox[0].min = xmin;
  bbox[0].max = xmax;
  bbox[1].min = ymin;
  bbox[1].max = ymax;
  bbox[2].min = zmin;
  bbox[2].max = zmax;
  #endif



  base::StateSpacePtr space(new base::RealVectorStateSpace(3));
  base::RealVectorBounds bounds(3);
  for (size_t i = 0; i < 3; ++i) {
    bounds.setLow(i, bbox[i].min);
    bounds.setHigh(i, bbox[i].max);
  }
  space->as<base::RealVectorStateSpace>()->setBounds(bounds);

  base::SpaceInformationPtr si(new base::SpaceInformation(space));
  base::StateValidityCheckerPtr stateValidityChecker(new MyStateValidityChecker(si, environment, robot));
  si->setStateValidityChecker(stateValidityChecker);
  base::MotionValidatorPtr motionValidator(new MyMotionValidator(si, environment, robot));
  si->setMotionValidator(motionValidator);
  // si->setStateValidityCheckingResolution(0.03); // 3%; TODO: remove and add proper motion validator
  si->setup();

  base::ProblemDefinitionPtr pdef(new base::ProblemDefinition(si));
  base::ScopedState<base::RealVectorStateSpace> start(space);
  // fill start state
  base::ScopedState<base::RealVectorStateSpace> goal(space);
  // fill goal state
  pdef->setStartAndGoalStates(start, goal);

  geometric::SPARS p(si);
  p.setProblemDefinition(pdef);
  std::cout << p.getDenseDeltaFraction()
  << " " << p.getSparseDeltaFraction()
  << " " << p.getStretchFactor()
  << " " << p.getMaxFailures() << std::endl;
  p.setSparseDeltaFraction(0.1);
  p.constructRoadmap(base::timedPlannerTerminationCondition(30), true);

  const auto& roadmap = p.getRoadmap();
  auto stateProperty = boost::get(geometric::SPARS::vertex_state_t(), roadmap);

  std::cout << "#vertices " << boost::num_vertices(roadmap) << " #edges " << boost::num_edges(roadmap) << std::endl;
#if 0
  // Write output file
  YAML::Emitter out;
  out.SetIndent(2);

  out << YAML::BeginMap;

  // vertices
  out << YAML::Key << "vertices";
  out << YAML::Value << YAML::BeginSeq;

  for (size_t i = 0; i < boost::num_vertices(roadmap); ++i) {
    base::State* state = stateProperty[i];
    if (state) {
      const base::RealVectorStateSpace::StateType* typedState = state->as<base::RealVectorStateSpace::StateType>();
      // std::cout << state << std::endl;

      out << YAML::BeginMap;
      out << YAML::Key << "name";
      out << YAML::Value << "v" + std::to_string(i);
      out << YAML::Key << "pos";
      out << YAML::Value << YAML::Flow << YAML::BeginSeq << (*typedState)[0] << (*typedState)[1] << (*typedState)[2] << YAML::EndSeq;
      out << YAML::EndMap;
    }
  }
  out << YAML::EndSeq;

  // edges
  out << YAML::Key << "edges";
  out << YAML::Value << YAML::BeginSeq;

  size_t i = 0;
  BOOST_FOREACH (const geometric::SPARStwo::Edge e, boost::edges(roadmap)) {
    out << YAML::BeginMap;
    out << YAML::Key << "name";
    out << YAML::Value << "e" + std::to_string(i);
    out << YAML::Key << "from";
    out << YAML::Value << "v" + std::to_string(boost::source(e, roadmap));
    out << YAML::Key << "to";
    out << YAML::Value << "v" + std::to_string(boost::target(e, roadmap));
    out << YAML::EndMap;

    ++i;
  }
  out << YAML::EndSeq;
  out << YAML::EndMap;

  std::ofstream ofstr("roadmap.yaml");
  ofstr << out.c_str();
#endif


#if 1
  struct edge {
    edge(size_t from, size_t to)
      : from(from)
      , to(to)
    {
    }

    size_t from;
    size_t to;
  };

  std::vector<position_t> vertices;
  std::vector<edge> edges;

  std::vector<std::string> vertexNames;

  if (addVerticesFile.size() > 0) {
    YAML::Node config = YAML::LoadFile(addVerticesFile);

    for (const auto& node : config["vertices"]) {
      const auto& pos = node["pos"];
      position_t p(
        pos[0].as<float>(),
        pos[1].as<float>(),
        pos[2].as<float>());
      std::string name = node["name"].as<std::string>();

      fcl::Transform3f env_tf;
      fcl::Transform3f robot_tf(toVec3f(p));
      CollisionRequest request;
      CollisionResult result;
      bool collision = false;

      if (environment && robot) {
        collide(environment, env_tf, robot, robot_tf, request, result);
        collision = result.isCollision();
      }

      if (!collision) {
        vertices.push_back(p);

        vertexNames.push_back(name);
      } else {
        std::cout << "Warning: additional vertex " << name << " is in collision!" << std::endl;
        // std::cout << result.getContact(0).pos << " " << p.x() << std::endl;
      }
    }
  }

  // add OMPL vertices
  for (size_t i = 0; i < boost::num_vertices(roadmap); ++i) {
    base::State* state = stateProperty[i];
    if (state) {
      const base::RealVectorStateSpace::StateType* typedState = state->as<base::RealVectorStateSpace::StateType>();
      vertices.push_back(position_t((*typedState)[0], (*typedState)[1], (*typedState)[2]));
    } else {
      vertices.push_back(position_t(-1, -1, -1));
    }
  }

  // add OMPL edges
  BOOST_FOREACH (const geometric::SPARS::SparseEdge e, boost::edges(roadmap)) {
    size_t i = boost::source(e, roadmap);
    size_t j = boost::target(e, roadmap);

    edges.push_back(edge(i + vertexNames.size(), j + vertexNames.size()));
  }

  // add additional edges

  typedef KDTreeVectorOfVectorsAdaptor< std::vector<position_t>, float, 3> kd_tree_t;
  kd_tree_t kd_tree(3, vertices);
  kd_tree.index->buildIndex();



  std::unordered_map<size_t, std::unordered_set<size_t> > edgeMap;

  for (size_t i = 0; i < vertexNames.size(); ++i) {
#if 0
      std::vector<std::pair<size_t, float> > ret_matches;

      nanoflann::SearchParams params;
      /*const size_t nMatches =*/ kd_tree.index->radiusSearch(&(vertices[i])[0], searchRadius, ret_matches, params);

      // totalNeighbors += ret_matches.size() - 1;

      for (const auto& match : ret_matches) {
        size_t j = match.first;
#else
      size_t num_results = 7;
      std::vector<size_t>   ret_index(num_results);
      std::vector<float> out_dist_sqr(num_results);
      num_results = kd_tree.index->knnSearch(&(vertices[i])[0], num_results, &ret_index[0], &out_dist_sqr[0]);
      ret_index.resize(num_results);
      out_dist_sqr.resize(num_results);
      for (size_t j : ret_index) {
#endif

      if (i != j) {
        // edges are unidirectional, so avoid adding any existing edge
        if (edgeMap[j].find(i) == edgeMap[j].end()) {
          Transform3f env_tf_beg;//(Translation3f(position_t(0, 0, 0)));
          Transform3f env_tf_end;//(Translation3f(position_t(0, 0, 0)));
          Transform3f robot_tf_beg(toVec3f(vertices[i]));
          Transform3f robot_tf_end(toVec3f(vertices[j]));

          ContinuousCollisionRequest request;//(10, 0.0001, fcl::CCDM_SCREW,
                      // fcl::GST_LIBCCD, fcl::CCDC_CONSERVATIVE_ADVANCEMENT);
          ContinuousCollisionResult result;
          bool collision = false;

          if (environment && robot) {
            continuousCollide(environment, env_tf_beg, env_tf_end,
                              robot, robot_tf_beg, robot_tf_end,
                              request, result);
            collision = result.is_collide;
          }

          if (!collision) {
            edges.push_back(edge(i, j));
            edgeMap[i].insert(j);
          }
        }
      }
    }

    // cout << "radiusSearch(): radius=" << search_radius << " -> " << nMatches << " matches\n";
    // for (size_t i=0;i<nMatches;i++) {
    //   cout << "idx["<< i << "]=" << ret_matches[i].first << " dist["<< i << "]=" << ret_matches[i].second << endl;
    // }
    // cout << "\n";
  }

  // std::cout << "Avg. possible neighbors: " << totalNeighbors / (double)vertices.size() << std::endl;
  // std::cout << "# duplicates: " << totalDuplicates << std::endl;
  // std::cout << "# vertices: " << vertices.size() - totalDuplicates << " #edges: " << edges.size() << std::endl;
#endif

  delete environment;
  delete robot;



  // Write output file
  YAML::Emitter out;
  out.SetIndent(2);

  out << YAML::BeginMap;

  // vertices
  out << YAML::Key << "vertices";
  out << YAML::Value << YAML::BeginSeq;

  for (size_t i = 0; i < vertices.size(); ++i) {
    out << YAML::BeginMap;
    out << YAML::Key << "name";
    out << YAML::Value << getVertexName(i, vertexNames);
    out << YAML::Key << "pos";
    out << YAML::Value << YAML::Flow << YAML::BeginSeq << vertices[i].x() << vertices[i].y() << vertices[i].z() << YAML::EndSeq;
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  // edges
  out << YAML::Key << "edges";
  out << YAML::Value << YAML::BeginSeq;

  for (size_t i = 0; i < edges.size(); ++i) {
    // out << YAML::Anchor("e" + std::to_string(i));
    out << YAML::BeginMap;
    out << YAML::Key << "name";
    out << YAML::Value << "e" + std::to_string(i);
    out << YAML::Key << "from";
    // out << YAML::Value << YAML::Alias("v" + std::to_string(edges[i].from));//edges[i].from;
    out << YAML::Value << getVertexName(edges[i].from, vertexNames);
    out << YAML::Key << "to";
    // out << YAML::Value << YAML::Alias("v" + std::to_string(edges[i].to));//edges[i].to;
    out << YAML::Value << getVertexName(edges[i].to, vertexNames);
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  out << YAML::EndMap;

  ofstream ofstr(outputFile.c_str());
  ofstr << out.c_str();

  return 0;
}

