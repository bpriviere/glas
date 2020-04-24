#include <cmath>
#include <vector>

#include "AVO.h"

#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

const float AVO_TWO_PI = 6.283185307179586f;

bool haveReachedGoals(const AVO::Simulator &simulator,
                      const std::vector<AVO::Vector2> &goals) {
  for (std::size_t i = 0; i < simulator.getNumAgents(); ++i) {
    if (AVO::absSq(simulator.getAgentPosition(i) - goals[i]) > 0.05 * 0.05f) {
      return false;
    }
  }

  return true;
}

int main(int argc, char** argv)
{
  std::string inputFile;

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(),"input file (YAML)")
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

  AVO::Simulator sim;

  sim.setTimeStep(0.1f);

  sim.setAgentDefaults(
    /* neighborDist*/ 1.0f,
    /* maxNeighbors*/ 30,
    /* timeHorizon*/ 10.0f,
    /* radius*/ 0.2f,
    /* maxSpeed*/ 0.5f,
    /* maxAccel*/ 1.0f,
    /* accelInterval*/ 0.2f);

  std::vector<AVO::Vector2> goals;

  // Load from file

  YAML::Node config = YAML::LoadFile(inputFile);
  for (const auto& node : config["agents"]) {
    const auto& start = node["start"];
    const auto& goal = node["goal"];

    AVO::Vector2 startPos(start[0].as<int>() + 0.5f, start[1].as<int>()+ 0.5f);
    sim.addAgent(startPos);

    AVO::Vector2 goalPos(goal[0].as<int>()+ 0.5f, goal[1].as<int>()+ 0.5f);
    goals.push_back(goalPos);

  }

  // const int numAgents = 4;
  // const float radius = 10;

  // for (std::size_t i = 0; i < numAgents; ++i) {
  //   const AVO::Vector2 position =
  //       radius * AVO::Vector2(std::cos(i * AVO_TWO_PI / numAgents),
  //                             std::sin(i * AVO_TWO_PI / numAgents));

  //   sim.addAgent(position);
  //   goals.push_back(-position);
  // }

  std::ofstream output("avo.csv");
  output << "t";
  for (size_t i = 0; i < sim.getNumAgents(); ++i) {
    output << ",x" << i << ",y" << i << ",vx" << i << ",vy" << i;
  }
  output << std::endl;

  int stayingAtGoal = 0;
  for (size_t i = 0; i < 1000 && stayingAtGoal < 20; ++i) {
    // output current simulation result
    output << sim.getGlobalTime();
    for (size_t i = 0; i < sim.getNumAgents(); ++i) {
      auto pos = sim.getAgentPosition(i);
      auto vel = sim.getAgentVelocity(i);
      output << "," << pos.getX() << "," << pos.getY() << "," << vel.getX() << "," << vel.getY();
    }
    output << std::endl;

    // set preferred velocities
    for (std::size_t i = 0; i < sim.getNumAgents(); ++i) {
      AVO::Vector2 toGoal = goals[i] - sim.getAgentPosition(i);

      if (AVO::absSq(toGoal) > 1.0f) {
        toGoal = normalize(toGoal);
      }

      sim.setAgentPrefVelocity(i, toGoal);

      // Perturb a little to avoid deadlocks due to perfect symmetry.
      float angle = std::rand() * 2.0f * M_PI / RAND_MAX;
      float dist = std::rand() * 0.1f / RAND_MAX;

      sim.setAgentPrefVelocity(i, sim.getAgentPrefVelocity(i) +
                              dist * AVO::Vector2(std::cos(angle), std::sin(angle)));
    }

    sim.doStep();

    if (haveReachedGoals(sim, goals)) {
      stayingAtGoal++;
    } else {
      stayingAtGoal = 0;
    }
  }

  return 0;
}
