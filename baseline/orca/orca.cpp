#include <cmath>
#include <cstddef>
#include <vector>

#include <fstream>
#include <random>
#include <iostream>

#include <boost/program_options.hpp>

#include <yaml-cpp/yaml.h>

#include "RVO.h"

/* Store the goals of the agents. */
std::vector<RVO::Vector2> goals;


void setupScenario(RVO::RVOSimulator *sim, const std::string& inputFile, float Rsense, float robotRadius)
{
  /* Specify the global time step of the simulation. */
  sim->setTimeStep(0.05f);

  /* Specify the default parameters for agents that are subsequently added. */
  sim->setAgentDefaults(
    /* neighborDist*/ Rsense,
    /* maxNeighbors*/ 5,
    /* timeHorizon*/ 1.0f,
    /* timeHorizonObst*/ 1.0f,
    /* radius*/ robotRadius,
    /* maxSpeed*/ 0.5f);


// Ring Example 
  /*
   * Add agents, specifying their start position, and store their goals on the
   * opposite side of the environment.
   */
//   for (size_t i = 0; i < numAgents; ++i) {
//     sim->addAgent(size *
//                   RVO::Vector2(std::cos(i * 2.0f * M_PI / numAgents),
//                                std::sin(i * 2.0f * M_PI / numAgents)));
//     goals.push_back(-sim->getAgentPosition(i));
//   }
// }


// // Random Examples 
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<> dis(-size, size);
//   for (size_t i = 0; i < numAgents;) {
//     float x = dis(gen);
//     float y = dis(gen);
//     RVO::Vector2 pos(x, y);
//     bool collision = false;
//     for (size_t j = 0; j < i; ++j) {
//       float dist = RVO::abs(pos - sim->getAgentPosition(j));
//       if (dist <= 3.5) {
//         collision = true;
//         break;
//       }
//     }
//     if (!collision) {
//       sim->addAgent(pos);
//       // find a collision-free goal
//       do {
//         RVO::Vector2 goal(dis(gen), dis(gen));
//         collision = false;
//         for (size_t j = 0; j < i; ++j) {
//           float dist = RVO::abs(goal - goals[j]);
//           if (dist <= 3.5) {
//             collision = true;
//             break;
//           }
//         }
//         if (!collision) {
//           goals.push_back(goal);
//           break;
//         }
//       } while(true);
//       // next agent
//       ++i;
//     }
//   }


    // Load from file

    YAML::Node config = YAML::LoadFile(inputFile);
    for (const auto& node : config["agents"]) {
      const auto& start = node["start"];
      const auto& goal = node["goal"];

      RVO::Vector2 startPos(start[0].as<float>() + 0.5f, start[1].as<float>()+ 0.5f);
      sim->addAgent(startPos);

      RVO::Vector2 goalPos(goal[0].as<float>()+ 0.5f, goal[1].as<float>()+ 0.5f);
      goals.push_back(goalPos);

    }

    /*
     * Add (polygonal) obstacles, specifying their vertices in counterclockwise
     * order.
     */
    for (const auto& node : config["map"]["obstacles"]) {
      float x = node[0].as<int>();
      float y = node[1].as<int>();
      std::vector<RVO::Vector2> obstacle;
      obstacle.push_back(RVO::Vector2(x, y));
      obstacle.push_back(RVO::Vector2(x+1, y));
      obstacle.push_back(RVO::Vector2(x+1, y+1));
      obstacle.push_back(RVO::Vector2(x, y+1));

      sim->addObstacle(obstacle);
    }

    int dimx = config["map"]["dimensions"][0].as<int>();
    int dimy = config["map"]["dimensions"][1].as<int>();
    {
      std::vector<RVO::Vector2> obstacle;
      obstacle.push_back(RVO::Vector2(-1, -1));
      obstacle.push_back(RVO::Vector2(0, -1));
      obstacle.push_back(RVO::Vector2(0, dimy));
      obstacle.push_back(RVO::Vector2(-1, dimy));
      sim->addObstacle(obstacle);
    }
    {
      std::vector<RVO::Vector2> obstacle;
      obstacle.push_back(RVO::Vector2(dimx, -1));
      obstacle.push_back(RVO::Vector2(dimx+1, -1));
      obstacle.push_back(RVO::Vector2(dimx+1, dimy));
      obstacle.push_back(RVO::Vector2(dimx, dimy));
      sim->addObstacle(obstacle);
    }
    {
      std::vector<RVO::Vector2> obstacle;
      obstacle.push_back(RVO::Vector2(0, -1));
      obstacle.push_back(RVO::Vector2(dimx+1, -1));
      obstacle.push_back(RVO::Vector2(dimx+1, 0));
      obstacle.push_back(RVO::Vector2(0, 0));
      sim->addObstacle(obstacle);
    }
    {
      std::vector<RVO::Vector2> obstacle;
      obstacle.push_back(RVO::Vector2(0, dimy));
      obstacle.push_back(RVO::Vector2(dimx+1, dimy));
      obstacle.push_back(RVO::Vector2(dimx+1, dimy+1));
      obstacle.push_back(RVO::Vector2(0, dimy+1));
      sim->addObstacle(obstacle);
    }

    /* Process the obstacles so that they are accounted for in the simulation. */
    sim->processObstacles();
}

void setPreferredVelocities(RVO::RVOSimulator *sim)
{
  /*
   * Set the preferred velocity to be a vector of unit magnitude (speed) in the
   * direction of the goal.
   */
  for (int i = 0; i < static_cast<int>(sim->getNumAgents()); ++i) {
    RVO::Vector2 goalVector = goals[i] - sim->getAgentPosition(i);

    if (RVO::absSq(goalVector) > 1.0f) {
      goalVector = RVO::normalize(goalVector);
    }

    sim->setAgentPrefVelocity(i, goalVector);

#if 0
    /*
     * Perturb a little to avoid deadlocks due to perfect symmetry.
     */
    float angle = std::rand() * 0.01f * M_PI / RAND_MAX;
    float dist = std::rand() * 0.01f / RAND_MAX;

    sim->setAgentPrefVelocity(i, sim->getAgentPrefVelocity(i) +
                              dist * RVO::Vector2(std::cos(angle), std::sin(angle)));
#endif
  }
}

bool reachedGoal(RVO::RVOSimulator *sim)
{
  /* Check if all agents have reached their goals. */
  for (size_t i = 0; i < sim->getNumAgents(); ++i) {
    if (RVO::absSq(sim->getAgentPosition(i) - goals[i]) > 0.05 * 0.05) { //sim->getAgentRadius(i) * sim->getAgentRadius(i)) {
      return false;
    }
  }

  return true;
}

int main(int argc, char** argv)
{
  std::string inputFile, outputFile;
  float Rsense;
  float robotRadius;

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(),"input file (YAML)")
    ("output,o", po::value<std::string>(&outputFile)->required(),"output file (csv)")
    ("Rsense", po::value<float>(&Rsense)->default_value(3.0),"sensing radius in meter")
    ("robotRadius", po::value<float>(&robotRadius)->default_value(0.16f),"radius of robots")
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

  /* Create a new simulator instance. */
  RVO::RVOSimulator sim;

  /* Set up the scenario. */
  setupScenario(&sim, inputFile, Rsense, robotRadius);

  std::ofstream output(outputFile);
  output << "t";
  for (size_t i = 0; i < sim.getNumAgents(); ++i) {
    output << ",x" << i << ",y" << i << ",vx" << i << ",vy" << i;
  }
  output << std::endl;

  /* Perform (and manipulate) the simulation. */
  for (size_t i = 0; i < 1000 && !reachedGoal(&sim); ++i) {
    // output current simulation result
    output << sim.getGlobalTime();
    for (size_t i = 0; i < sim.getNumAgents(); ++i) {
      auto pos = sim.getAgentPosition(i);
      auto vel = sim.getAgentVelocity(i);
      output << "," << pos.x() << "," << pos.y() << "," << vel.x() << "," << vel.y();
    }
    output << std::endl;

    setPreferredVelocities(&sim);
    sim.doStep();
  }

  // keep simulation running a bit longer
  for (size_t i = 0; i < 10; ++i) {
    // output current simulation result
    output << sim.getGlobalTime();
    for (size_t i = 0; i < sim.getNumAgents(); ++i) {
      auto pos = sim.getAgentPosition(i);
      auto vel = sim.getAgentVelocity(i);
      output << "," << pos.x() << "," << pos.y() << "," << vel.x() << "," << vel.y();
    }
    output << std::endl;

    setPreferredVelocities(&sim);
    // sim.doStep();    
  }

  return 0;
}
