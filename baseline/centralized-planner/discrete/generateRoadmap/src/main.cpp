#include <iostream>
#include <unordered_map>
#include <unordered_set>

// Boost
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

// Eigen
#include <Eigen/Core>

// FCL Headers
#include <fcl/collision.h>
#include <fcl/collision_node.h>
#include <fcl/traversal/traversal_node_setup.h>
#include <fcl/continuous_collision.h>

// Yaml
#include <yaml-cpp/yaml.h>

// Nanoflann
#include <nanoflann.hpp>
#include "KDTreeVectorOfVectorsAdaptor.h"

// OMPL
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/prm/SPARS.h>
#include <ompl/geometric/planners/prm/SPARStwo.h>

// local includes
#include "libCommon/common.h"
#include "libCommon/searchgraph.h"
#include "libCommon/Timer.hpp"

#include "fclHelper.h"
#include "NanoFlannSearchgraphAdaptor.h"
#include "fclStateValidityChecker.hpp"
#include "fclMotionValidator.hpp"

using namespace std;
using namespace fcl;
using namespace nanoflann;
using namespace ompl;

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

struct GenerationType {
  enum Type
  {
    Grid,
    Halton,
    SPARS,
    SPARS2,
  };
};

class HaltonStateSampler : public ompl::base::StateSampler
{
public:
  HaltonStateSampler(
    const ompl::base::StateSpace *space)
    : ompl::base::StateSampler(space)
    , m_i(0)
  {
    const ompl::base::RealVectorBounds &bounds = static_cast<const ompl::base::RealVectorStateSpace *>(space_)->getBounds();
    auto diff = bounds.getDifference();

    m_scale = std::max(std::max(diff[0], diff[1]), diff[3]);
  }

  void sampleUniform(ompl::base::State *state) override
  {
    const ompl::base::RealVectorBounds &bounds = static_cast<const ompl::base::RealVectorStateSpace *>(space_)->getBounds();
    while (true) {
      ++m_i;
      double x = halton(m_i, 2) * m_scale + bounds.low[0];
      if (bounds.low[0] <= x && bounds.high[0] >=x) {
        double y = halton(m_i, 3) * m_scale + bounds.low[1];
        if (bounds.low[1] <= y && bounds.high[1] >=y) {
          double z = halton(m_i, 5) * m_scale + bounds.low[2];
          if (bounds.low[2] <= z && bounds.high[2] >=z) {
            ompl::base::RealVectorStateSpace::StateType *rstate = static_cast<ompl::base::RealVectorStateSpace::StateType *>(state);
            rstate->values[0] = x;
            rstate->values[1] = y;
            rstate->values[2] = z;
            break;
          }
        }
      }
    }
  }

  void sampleUniformNear(ompl::base::State *state, const ompl::base::State *near, const double distance) override
  {
    throw std::runtime_error("sampleUniformNear not implemented!");
  }

  void sampleGaussian(ompl::base::State *state, const ompl::base::State *mean, const double stdDev) override
  {
    throw std::runtime_error("sampleGaussian not implemented!");
  }
private:
  int m_i;
  double m_scale;
};

ompl::base::StateSamplerPtr allocHaltonStateSampler(const ompl::base::StateSpace* space)
{
  return ompl::base::StateSamplerPtr(new HaltonStateSampler(space));
}

std::istream& operator>>(std::istream& in, GenerationType::Type& type)
{
    std::string token;
    in >> token;
    if (token == "Grid")
        type = GenerationType::Grid;
    else if (token == "Halton")
        type = GenerationType::Halton;
    else if (token == "SPARS")
        type = GenerationType::SPARS;
    else if (token == "SPARS2")
        type = GenerationType::SPARS2;
    else
        in.setstate(std::ios_base::failbit);
    return in;
}

bool checkBounds(
  const std::array<fclHelper::range, 3>& bbox,
  const position_t& pos)
{
  return pos.x() >= bbox[0].min
      && pos.x() <= bbox[0].max
      && pos.y() >= bbox[1].min
      && pos.y() <= bbox[1].max
      && pos.z() >= bbox[2].min
      && pos.z() <= bbox[2].max;
}

int main(int argc, char** argv) {


  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string environmentFile;
  std::string robotFile;
  std::string addVerticesFile;
  std::string outputFile;
  std::string configFile;
  // size_t numPoints;
  // double searchRadius;
  double xmin,xmax,ymin,ymax,zmin,zmax;
  GenerationType::Type genType = GenerationType::Grid;
  size_t dimension;
  float fixedZ;
  desc.add_options()
      ("help", "produce help message")
      ("environment,e", po::value<std::string>(&environmentFile)->required(), "input file for environment (STL)")
      ("robot,r", po::value<std::string>(&robotFile)->required(), "input file for robot (STL)")
      ("addVertices,a", po::value<std::string>(&addVerticesFile), "input file for additional vertices (YAML)")
      ("output,o", po::value<std::string>(&outputFile)->required(), "output file for graph (YAML)")
      ("config,c", po::value<std::string>(&configFile)->required(), "config file for advanced parameters (YAML)")
      ("type", po::value<GenerationType::Type>(&genType)->default_value(GenerationType::Grid)->multitoken(), "Method, one of [Grid,Halton,SPARS,SPARS2]. Default: Grid")
      ("dimension", po::value<size_t>(&dimension)->default_value(3), "roadmap dimension (2 or 3); For 2 the z-coordinate used is fixedZ")
      ("fixedZ", po::value<float>(&fixedZ)->default_value(0.0), "z-coordinate used if dimension is set to 2")
      // ("numPoints", po::value<size_t>(&numPoints)->default_value(100), "Number of points to sample in bounding box")
      // ("searchRadius", po::value<double>(&searchRadius)->default_value(0.5), "kd tree search radius")
      ("xmin", po::value<double>(&xmin), "bounding box minimum x")
      ("xmax", po::value<double>(&xmax), "bounding box maximum x")
      ("ymin", po::value<double>(&ymin), "bounding box minimum y")
      ("ymax", po::value<double>(&ymax), "bounding box maximum y")
      ("zmin", po::value<double>(&zmin), "bounding box minimum z")
      ("zmax", po::value<double>(&zmax), "bounding box maximum z")
  ;

  po::variables_map vm;
  try
  {
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

  YAML::Node cfg = YAML::LoadFile(configFile);

  CollisionGeometry* environment = fclHelper::createCollisionGeometryFromFile(environmentFile);
  CollisionGeometry* robot = fclHelper::createCollisionGeometryFromFile(robotFile);

  std::array<fclHelper::range, 3> bbox = fclHelper::boundingBox(environment);
  if (vm.count("xmin")) bbox[0].min = xmin;
  if (vm.count("xmax")) bbox[0].max = xmax;
  if (vm.count("ymin")) bbox[1].min = ymin;
  if (vm.count("ymax")) bbox[1].max = ymax;
  if (vm.count("zmin")) bbox[2].min = zmin;
  if (vm.count("zmax")) bbox[2].max = zmax;

  fcl::Transform3f env_tf;//(Vec3f(-0.5, -0.5, -0.5));
  for (size_t i = 0; i < 3; ++i) {
    bbox[i].min += env_tf.getTranslation()[i];
    bbox[i].max += env_tf.getTranslation()[i];
  }

  std::cout << "Bounding box (environment): " << std::endl;
  std::cout << bbox[0].min << "," << bbox[0].max << std::endl;
  std::cout << bbox[1].min << "," << bbox[1].max << std::endl;
  std::cout << bbox[2].min << "," << bbox[2].max << std::endl;

  std::array<fclHelper::range, 3> bbox2 = fclHelper::boundingBox(robot);
  std::cout << "Bounding box (robot): " << std::endl;
  std::cout << bbox2[0].min << "," << bbox2[0].max << std::endl;
  std::cout << bbox2[1].min << "," << bbox2[1].max << std::endl;
  std::cout << bbox2[2].min << "," << bbox2[2].max << std::endl;

  // adjust bounding box
  for (size_t i = 0; i < 3; ++i) {
    bbox[i].min -= bbox2[i].min;
    bbox[i].max -= bbox2[i].max;
  }

  Timer timer;
  searchGraph_t roadmap;
  std::set<std::string> addVerticesNames;

  size_t addEdgesForVertices = 0;
  if (addVerticesFile.size() > 0) {
    YAML::Node config = YAML::LoadFile(addVerticesFile);

    for (const auto& node : config["vertices"]) {
      const auto& pos = node["pos"];
      position_t p(
        pos[0].as<float>(),
        pos[1].as<float>(),
        pos[2].as<float>());
      std::string name = node["name"].as<std::string>();
      addVerticesNames.insert(name);

      fcl::Transform3f robot_tf(toVec3f(p));
      CollisionRequest request;
      CollisionResult result;
      bool collision = false;

      if (environment && robot) {
        collide(environment, env_tf, robot, robot_tf, request, result);
        collision = result.isCollision();
      }

      collision |= !checkBounds(bbox, p);

      if (!collision) {
        auto v = boost::add_vertex(roadmap);
        roadmap[v].pos = p;
        roadmap[v].name = name;
      } else {
        std::cout << "Warning: additional vertex " << name << " is in collision!" << std::endl;
        // std::cout << result.getContact(0).pos << " " << p.x() << std::endl;
      }
    }
  }

  if (genType == GenerationType::Grid) {
    auto node = cfg["grid"];
    const float gridSize = node["gridSize"].as<float>();
    const float xOffset = node["xOffset"].as<float>();
    const float yOffset = node["yOffset"].as<float>();
    const float zOffset = node["zOffset"].as<float>();
    float zmin = bbox[2].min + bbox2[2].min + gridSize/2;
    float zmax = bbox[2].max;
    if (dimension == 2) {
      zmin = fixedZ;
      zmax = fixedZ + gridSize / 2.0;
    }

    for (float x = bbox[0].min + bbox2[0].min + gridSize/2 + xOffset; x < bbox[0].max; x += gridSize) {
      for (float y = bbox[1].min + bbox2[1].min + gridSize/2 + yOffset; y < bbox[1].max; y += gridSize) {
        for (float z = zmin + zOffset; z < zmax; z += gridSize) {
          fcl::Transform3f robot_tf(Vec3f(x, y, z));
          CollisionRequest request;
          CollisionResult result;
          bool collision = false;

          // CollisionObject<float> obj1(environment, env_tf);
          // CollisionObject<float> obj2(robot, robot_tf);
          if (environment && robot) {
            collide(environment, env_tf, robot, robot_tf, request, result);
            collision = result.isCollision();
          }

          if (!collision) {
            auto v = boost::add_vertex(roadmap);
            roadmap[v].pos = position_t(x, y, z);
            roadmap[v].name = "v" + std::to_string(v);
          }
        }
      }
    }
    // add edges for all vertices
    addEdgesForVertices = num_vertices(roadmap);
  }
  else if (genType == GenerationType::Halton) {
    auto node = cfg["halton"];
    const size_t numPoints = node["numPoints"].as<size_t>();
    const double scale = std::max(std::max(bbox[0].max - bbox[0].min, bbox[1].max - bbox[1].min), bbox[2].max - bbox[2].min);
    size_t numSamples = 0;
    for (size_t i = 0; ;++i) {
      double x = halton(i+1, 2) * scale + bbox[0].min;
      if (bbox[0].min <= x && bbox[0].max >=x) {
        double y = halton(i+1, 3) * scale + bbox[1].min;
        if (bbox[1].min <= y && bbox[1].max >=y) {
          double z = halton(i+1, 5) * scale + bbox[2].min;
          if (bbox[2].min <= z && bbox[2].max >=z) {
            // std::cout << x << "," << y << "," << z << std::endl;
            // position_t pos(x, y, z);

            fcl::Transform3f env_tf; //(Vec3f(0, 0, 0));
            fcl::Transform3f robot_tf(Vec3f(x, y, z));
            CollisionRequest request;
            CollisionResult result;

            // CollisionObject<float> obj1(environment, env_tf);
            // CollisionObject<float> obj2(robot, robot_tf);

            collide(environment, env_tf, robot, robot_tf, request, result);

            if (!result.isCollision()) {
              auto v = boost::add_vertex(roadmap);
              roadmap[v].pos = position_t(x, y, z);
              roadmap[v].name = "v" + std::to_string(v);
            }

            ++numSamples;
            if (numSamples == numPoints) {
              break;
            }
          }
        }
      }
    }

    // add edges for all vertices
    addEdgesForVertices = num_vertices(roadmap);
  }

  else if (genType == GenerationType::SPARS
        || genType == GenerationType::SPARS2) {
    // add edges for additional vertices, only
    addEdgesForVertices = num_vertices(roadmap);

    base::StateSpacePtr space(new base::RealVectorStateSpace(dimension));
    base::RealVectorBounds bounds(dimension);
    for (size_t i = 0; i < dimension; ++i) {
      bounds.setLow(i, bbox[i].min);
      bounds.setHigh(i, bbox[i].max);
    }
    space->as<base::RealVectorStateSpace>()->setBounds(bounds);
    if (genType == GenerationType::SPARS) {
      auto node = cfg["spars"];
      if (node["sampler"].as<std::string>() == "halton") {
        space->setStateSamplerAllocator(allocHaltonStateSampler);
      }
    }

    base::SpaceInformationPtr si(new base::SpaceInformation(space));
    base::StateValidityCheckerPtr stateValidityChecker(new fclStateValidityChecker(si, environment, robot, env_tf));
    si->setStateValidityChecker(stateValidityChecker);
    base::MotionValidatorPtr motionValidator(new fclMotionValidator(si, environment, robot, env_tf));
    si->setMotionValidator(motionValidator);
    // si->setStateValidityCheckingResolution(0.03); // 3%; TODO: remove and add proper motion validator
    si->setup();

    base::ProblemDefinitionPtr pdef(new base::ProblemDefinition(si));
    base::ScopedState<base::RealVectorStateSpace> start(space);
    // fill start state
    base::ScopedState<base::RealVectorStateSpace> goal(space);
    // fill goal state
    pdef->setStartAndGoalStates(start, goal);

    if (genType == GenerationType::SPARS) {
      auto node = cfg["spars"];

      geometric::SPARS p(si);
      p.setProblemDefinition(pdef);
      // std::cout << p.getDenseDeltaFraction()
      // << "Delta: " << p.getSparseDeltaFraction() * si->getMaximumExtent()
      // << " " << p.getStretchFactor()
      // << " " << p.getMaxFailures() << std::endl;
      // p.setSparseDeltaFraction(1.0 / si->getMaximumExtent() );

      p.setDenseDeltaFraction(node["denseDelta"].as<float>() / si->getMaximumExtent() );
      p.setSparseDeltaFraction(node["sparseDelta"].as<float>() / si->getMaximumExtent() );
      p.setStretchFactor(node["stretchFactor"].as<float>());
      p.setMaxFailures(node["maxFailures"].as<int>());

      p.constructRoadmap(/*base::timedPlannerTerminationCondition(30)*/base::IterationTerminationCondition(node["maxIter"].as<int>()), true);

      p.printDebug();
#if 1
      const auto& roadmapOMPL = p.getRoadmap();
      // const auto& roadmapOMPL = p.getDenseGraph();
      auto stateProperty = boost::get(geometric::SPARS::vertex_state_t(), roadmapOMPL);

      std::unordered_map<geometric::SPARS::SparseVertex, vertex_t> vertexMap;
      for (size_t i = 0; i < boost::num_vertices(roadmapOMPL); ++i) {
        base::State* state = stateProperty[i];
        if (state) {
          position_t p (-1, -1, -1);
          const base::RealVectorStateSpace::StateType* typedState = state->as<base::RealVectorStateSpace::StateType>();
          if (dimension == 3) {
            p = position_t((*typedState)[0], (*typedState)[1], (*typedState)[2]);
          } else if (dimension == 2) {
            p = position_t((*typedState)[0], (*typedState)[1], fixedZ);
          }
          auto v = boost::add_vertex(roadmap);
          roadmap[v].pos = p;
          roadmap[v].name = "v" + std::to_string(v);
          vertexMap[i] = v;
        }
      }

      // add OMPL edges
      BOOST_FOREACH (const geometric::SPARS::SparseEdge e, boost::edges(roadmapOMPL)) {
        size_t i = boost::source(e, roadmapOMPL);
        size_t j = boost::target(e, roadmapOMPL);

        add_edge(vertexMap.at(i), vertexMap.at(j), roadmap);
      }
      #else
      const auto& roadmapOMPL = p.getDenseGraph();
      auto stateProperty = boost::get(geometric::SPARS::vertex_state_t(), roadmapOMPL);

      size_t offset = num_vertices(roadmap);
      for (size_t i = 0; i < boost::num_vertices(roadmapOMPL); ++i) {
        base::State* state = stateProperty[i];
        position_t p (-1, -1, -1);
        if (state) {
          const base::RealVectorStateSpace::StateType* typedState = state->as<base::RealVectorStateSpace::StateType>();
          p = position_t((*typedState)[0], (*typedState)[1], (*typedState)[2]);
        }
        auto v = boost::add_vertex(roadmap);
        roadmap[v].pos = p;
        roadmap[v].name = "v" + std::to_string(v);
      }

      // add OMPL edges
      BOOST_FOREACH (const geometric::SPARS::DenseEdge e, boost::edges(roadmapOMPL)) {
        size_t i = boost::source(e, roadmapOMPL);
        size_t j = boost::target(e, roadmapOMPL);

        add_edge(i + offset, j + offset, roadmap);
      }
#endif

    }

    if (genType == GenerationType::SPARS2) {
      auto node = cfg["spars2"];

      geometric::SPARStwo p(si);
      p.setProblemDefinition(pdef);

      p.setDenseDeltaFraction(node["denseDelta"].as<float>() / si->getMaximumExtent() );
      p.setSparseDeltaFraction(node["sparseDelta"].as<float>() / si->getMaximumExtent() );
      p.setStretchFactor(node["stretchFactor"].as<float>());
      p.setMaxFailures(node["maxFailures"].as<int>());

      p.constructRoadmap(/*base::timedPlannerTerminationCondition(30)*/base::IterationTerminationCondition(node["maxIter"].as<int>()), true);

      const auto& roadmapOMPL = p.getRoadmap();
      auto stateProperty = boost::get(geometric::SPARStwo::vertex_state_t(), roadmapOMPL);

      std::unordered_map<geometric::SPARS::SparseVertex, vertex_t> vertexMap;
      for (size_t i = 0; i < boost::num_vertices(roadmapOMPL); ++i) {
        base::State* state = stateProperty[i];
        if (state) {
          position_t p (-1, -1, -1);
          const base::RealVectorStateSpace::StateType* typedState = state->as<base::RealVectorStateSpace::StateType>();
          if (dimension == 3) {
            p = position_t((*typedState)[0], (*typedState)[1], (*typedState)[2]);
          } else if (dimension == 2) {
            p = position_t((*typedState)[0], (*typedState)[1], fixedZ);
          }
          auto v = boost::add_vertex(roadmap);
          roadmap[v].pos = p;
          roadmap[v].name = "v" + std::to_string(v);
          vertexMap[i] = v;
        }
      }

      // add OMPL edges
      BOOST_FOREACH (const geometric::SPARStwo::Edge e, boost::edges(roadmapOMPL)) {
        size_t i = boost::source(e, roadmapOMPL);
        size_t j = boost::target(e, roadmapOMPL);

        add_edge(vertexMap.at(i), vertexMap.at(j), roadmap);
      }
    }

  }


  typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<float, NanoFlannSearchgraphAdaptor>, NanoFlannSearchgraphAdaptor, 3 > kd_tree_t;
  NanoFlannSearchgraphAdaptor adaptor(roadmap);
  kd_tree_t index(3, adaptor);
  index.buildIndex();

  size_t totalNeighbors = 0;
  size_t totalDuplicates = 0;

  // find duplicates during the process
  // since the additional vertices are at the beginning, they should not be marked
  std::vector<bool> vertexDuplicates(num_vertices(roadmap), false);

  for (size_t i = 0; i < num_vertices(roadmap); ++i) {
    if (!vertexDuplicates[i]) {
      std::vector<std::pair<size_t, float> > ret_matches;

      nanoflann::SearchParams params;
      /*const size_t nMatches =*/ index.radiusSearch(&(roadmap[i].pos)[0], 1e-3, ret_matches, params);

      // totalNeighbors += ret_matches.size() - 1;

      for (const auto& match : ret_matches) {
        // only remove vertices that were not manually added
        if (i != match.first && addVerticesNames.find(roadmap[match.first].name) == addVerticesNames.end()) {
          vertexDuplicates[match.first] = true;
          ++totalDuplicates;
          // std::cout << "duplicate: " << getVertexName(match.first, vertexNames) << std::endl;
        }
      }
    }
  }

  std::unordered_map<size_t, std::unordered_set<size_t> > edgeMap;

  auto node = cfg["connection"];
  // bool radiusSearch = node["mode"].as<std::string>() == "radius";
  float radius = node["maxRadius"].as<float>();
  size_t maxNeighbors = node["maxNeighbors"].as<float>();


  for (size_t i = 0; i < addEdgesForVertices; ++i) {
    if (!vertexDuplicates[i]) {
#if 0
      std::vector<std::pair<size_t, float> > ret_matches;

      nanoflann::SearchParams params;
      /*const size_t nMatches =*/ index.radiusSearch(&(roadmap[i].pos)[0], searchRadius, ret_matches, params);

      // totalNeighbors += ret_matches.size() - 1;

      for (const auto& match : ret_matches) {
        size_t j = match.first;
#else
      size_t num_results = maxNeighbors + 1; // +1 because it finds itself, too
      std::vector<size_t>   ret_index(num_results);
      std::vector<float> out_dist_sqr(num_results);
      num_results = index.knnSearch(&(roadmap[i].pos)[0], num_results, &ret_index[0], &out_dist_sqr[0]);
      if (num_results == 0) {
        std::cout << "WARNING: Couldn't find any neighbors for " << roadmap[i].name << std::endl;
      }
      ret_index.resize(num_results);
      out_dist_sqr.resize(num_results);
      for (size_t k = 0; k < num_results; ++k) {
        size_t j = ret_index[k];
#endif
        if (i != j
            && !vertexDuplicates[j]
            && out_dist_sqr[k] < radius * radius) {

          ++totalNeighbors;

          // edges are unidirectional, so avoid adding any existing edge
          if (edgeMap[j].find(i) == edgeMap[j].end()) {
            Transform3f robot_tf_beg(toVec3f(roadmap[i].pos));
            Transform3f robot_tf_end(toVec3f(roadmap[j].pos));

            ContinuousCollisionRequest request;//(10, 0.0001, fcl::CCDM_SCREW,
                        // fcl::GST_LIBCCD, fcl::CCDC_CONSERVATIVE_ADVANCEMENT);
            ContinuousCollisionResult result;
            bool collision = false;

            if (environment && robot) {
              continuousCollide(environment, env_tf, env_tf,
                                robot, robot_tf_beg, robot_tf_end,
                                request, result);
              collision = result.is_collide;
            }

            if (!collision) {
              // edges.push_back(edge(i, j));
              add_edge(i, j, roadmap);
              edgeMap[i].insert(j);
            }
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

  timer.stop();
  std::cout << "Elapsed Time: " << timer.elapsedSeconds() << std::endl;

  std::cout << "Avg. possible neighbors: " << totalNeighbors / (double)num_vertices(roadmap) << std::endl;
  std::cout << "# duplicates: " << totalDuplicates << std::endl;
  std::cout << "# vertices: " << num_vertices(roadmap) - totalDuplicates << " #edges: " << num_edges(roadmap) << std::endl;


  delete environment;
  delete robot;

  // assign edge names
  size_t i = 0;
  BOOST_FOREACH (const edge_t e, boost::edges(roadmap)) {
    roadmap[e].name = "e" + std::to_string(i);
    ++i;
  }

  saveSearchGraph(roadmap, outputFile);

  return 0;
}

