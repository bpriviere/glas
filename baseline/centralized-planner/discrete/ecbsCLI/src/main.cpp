#include "libCommon/map.h"
#include "libEcbs/ecbs.h"
#include <string>
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <dlib/optimization/max_cost_assignment.h>

#include "boost/program_options.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;

#include <boost/graph/dijkstra_shortest_paths.hpp>

#include "libCommon/Timer.hpp"
#include "libCommon/AgentsLoaderYaml.h"
#include "libCommon/searchgraph.h"

// Yaml
#include "yaml-cpp/yaml.h"

using namespace std;

void shortestPath(
  const searchGraph_t& searchGraph,
  vertex_t source,
  std::vector<float>& distances)
{

  std::vector<vertex_t> predecessors(num_vertices(searchGraph));
  distances.resize(num_vertices(searchGraph));

  dijkstra_shortest_paths(searchGraph, source,
      weight_map(get(&Edge::length, searchGraph))
      .distance_map(make_iterator_property_map(distances.begin(),
                                               get(boost::vertex_index, searchGraph))));

  // dijkstra_shortest_paths(searchGraph, source,
  //                         predecessor_map(boost::make_iterator_property_map(predecessors.begin(), get(boost::vertex_index, searchGraph))).
  //                         distance_map(boost::make_iterator_property_map(distances.begin(), get(boost::vertex_index, searchGraph))),

  //                         );
}

// See "Assignment Problems", by Rainer Burkard , Mauro Dell'Amico and Silvano Martello (Threshold algorithm, p 175)
// TODO: This method is a brute-force method, faster version would use a binary search
void findMinMaxAssignment(
  const dlib::matrix<uint32_t>& cost,
  std::vector<long>& assignment)
{
  using namespace dlib;

  std::set<uint32_t> allCosts;
  for (size_t i = 0; i < assignment.size(); ++i) {
    for (size_t j = 0; j < assignment.size(); ++j) {
      allCosts.insert(cost(i, j));
    }
  }

  std::vector<uint32_t> costList(allCosts.begin(), allCosts.end());
  std::sort(costList.begin(), costList.end());

  uint32_t lastCost = 0;
  matrix<uint32_t> binaryCost(assignment.size(), assignment.size());
  for (size_t k = 0; k < costList.size(); ++k) {
    uint32_t maxCost = costList[k];
    if (maxCost > lastCost) {
      // std::cout << "attempt: " << maxCost << std::endl;
      for (size_t i = 0; i < assignment.size(); ++i) {
        for (size_t j = 0; j < assignment.size(); ++j) {
          if (cost(i, j) <= maxCost) {
            binaryCost(i,j) = 1;
          } else {
            binaryCost(i,j) = 0;
          }
        }
      }
      assignment = max_cost_assignment(binaryCost);
      // std::cout << "   " << assignment_cost(binaryCost, assignment) << std::endl;
      if (assignment_cost(binaryCost, assignment) >= assignment.size()) {
        // found it!
        break;
      }

      lastCost = maxCost;
    }
  }
}

// Hungarian method
void findMinSumAssignment(
  const dlib::matrix<uint32_t>& cost,
  std::vector<long>& assignment)
{
  using namespace dlib;
  assignment = max_cost_assignment(-cost);
}

enum AssignmentMethod
{
  AssignmentMinSum,
  AssignmentMinMax,
};

void findBestAssignment(
  const searchGraph_t& searchGraph,
  std::vector<Agent>& agents,
  AssignmentMethod method)
{
  using namespace dlib;

  // Compute cost matrix
  matrix<uint32_t> cost(agents.size(), agents.size());

  for (size_t i = 0; i < agents.size(); ++i) {
    std::vector<float> distances;
    shortestPath(
      searchGraph,
      agents[i].start,
      distances);
    for (size_t j = 0; j < agents.size(); ++j) {
      vertex_t goalVertex = agents[j].goal;
      uint32_t goalCost = (uint32_t)(distances[goalVertex] * 100);
      cost(i, j) = goalCost;
    }
  }

  std::vector<long> assignment;
  for (size_t i = 0; i < agents.size(); ++i) {
    assignment.push_back(i);
  }

  // Print stats about initial/given assignment
  std::cout << "Original Cost: " << assignment_cost(cost, assignment) / 100 << std::endl;

  uint32_t maxCost = 0;
  for (unsigned long i = 0; i < assignment.size(); ++i)
  {
    maxCost = std::max(maxCost, cost(i, assignment[i]));
  }
  std::cout << "max: " << maxCost / 100 << std::endl;

  // compute best assignment
  switch (method)
  {
  case AssignmentMinSum:
    findMinSumAssignment(cost, assignment);
    break;
  case AssignmentMinMax:
    findMinMaxAssignment(cost, assignment);
  }

  // Print stats about computed assignment
  std::cout << "Best Cost: " << assignment_cost(cost, assignment) / 100 << std::endl;

  maxCost = 0;
  for (unsigned long i = 0; i < assignment.size(); ++i)
  {
    maxCost = std::max(maxCost, cost(i, assignment[i]));
  }
  std::cout << "max: " << maxCost / 100 << std::endl;

  // actual re-assign agents
  std::vector<vertex_t> originalGoals;
  for (size_t i = 0; i < agents.size(); ++i) {
    originalGoals.push_back(agents[i].goal);
  }

  for (size_t i = 0; i < assignment.size(); ++i) {
    agents[i].goal = originalGoals[assignment[i]];
  }
}

int main(int argc, char** argv) {


  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string mapFile;
  std::string agentsFile;
  std::string highwayFile;
  std::string initialPathsFile;
  double highwayWeight;
  double focalWeight;
  bool tweakGVal;
  std::string outputFile;
  std::string statFile;
  bool minSumAssignment;
  bool minMaxAssignment;
  desc.add_options()
      ("help", "produce help message")
      ("map,m", po::value<std::string>(&mapFile)->required(), "input file for roadmap (YAML)")
      ("agents,a", po::value<std::string>(&agentsFile)->required(), "input file for agents")
      ("highway,w", po::value<std::string>(&highwayFile), "input file for highway")
      ("initialPaths", po::value<std::string>(&initialPathsFile), "input file for initial paths")
      ("highway-weight,g", po::value<double>(&highwayWeight)->default_value(1.0), "highway weight")
      ("focal-weight,f", po::value<double>(&focalWeight)->default_value(1.0), "focal weight")
      ("tweakGVal", po::bool_switch(&tweakGVal)->default_value(false), "tweak g value")
      ("output,o", po::value<std::string>(&outputFile)->required(), "output file for schedule")
      ("statFile", po::value<std::string>(&statFile), "output file for statistics")
      ("minSumAssignment", po::bool_switch(&minSumAssignment)->default_value(false), "reorder goals optimally using hungarian method (minimizes the sum)")
      ("minMaxAssignment", po::bool_switch(&minMaxAssignment)->default_value(false), "reorder goals optimally using Threshold algorithm (minimizes maximum cost)")
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

  // construct search graph
  searchGraph_t searchGraph;
  std::unordered_map<std::string, vertex_t> vNameToV;
  std::unordered_map<std::string, edge_t> eNameToE;
  loadSearchGraph(searchGraph, vNameToV, eNameToE, mapFile);


  std::vector< std::vector<uint32_t> > initialPaths;

  std::ifstream is(initialPathsFile.c_str());
  if (is.is_open()) {
    // read schedule
    pt::ptree tree;
    pt::read_json(initialPathsFile, tree);

    for (auto& item : tree.get_child("agents")) {
      initialPaths.resize(initialPaths.size() + 1);
      auto& path = initialPaths.back();
      for (auto& item2 : item.second.get_child("path")) {
          path.push_back(item2.second.get<uint32_t>("locationId"));
      }
    }
  }



  // std::ifstream is(initialPathsFile.c_str());
  // if (is.is_open()) {
  //   for (size_t i = 0; i < (size_t)al.num_of_agents; ++i) {
  //     std::string line;
  //     std::getline(is, line);

  //     boost::char_separator<char> sep(",");
  //     boost::tokenizer< boost::char_separator<char> > tok(line, sep);
  //     for (auto t = tok.begin(); t != tok.end(); ++t) {
  //       initialPaths[i].push_back(atoi(t->c_str()));
  //     }
  //   }
  // }

  // std::vector<Agent> agents;
  // for (size_t i = 0; i < al.numAgents(); ++i) {
  //   agents.push_back(Agent(
  //     al.name()[i],
  //     ml.positionToIdx(al.initialLocations()[i]),
  //     ml.positionToIdx(al.goalLocations()[assignment[i]])));
  // }

  // read agents
  std::vector<Agent> agents;

  YAML::Node agentsCfg = YAML::LoadFile(agentsFile);
  for (const auto& agent : agentsCfg["agents"]) {
    std::string name = agent["name"].as<std::string>();
    std::string type = agent["type"].as<std::string>();
    std::string startName = agent["start"].as<std::string>();
    auto iterStart = vNameToV.find(startName);
    if (iterStart == vNameToV.end()) {
      std::cerr << "ERROR! Could not find vertex " << startName << std::endl;
      continue;
    }
    std::string goalName = agent["goal"].as<std::string>();
    auto iterGoal = vNameToV.find(goalName);
    if (iterGoal == vNameToV.end()) {
      std::cerr << "ERROR! Could not find vertex " << goalName << std::endl;
      continue;
    }
    agents.push_back(Agent(name, type, iterStart->second, iterGoal->second));
  }

  Timer timerAssignment;

  if (minSumAssignment) {
    findBestAssignment(searchGraph, agents, AssignmentMinSum);
  }
  if (minMaxAssignment) {
    findBestAssignment(searchGraph, agents, AssignmentMinMax);
  }
  timerAssignment.stop();
  std::cout << "Assignment elapsed Time: " << timerAssignment.elapsedSeconds() << std::endl;

  Timer timer;
  ECBS ecbs = ECBS(
    &searchGraph,
    agents,
    highwayWeight,
    focalWeight,
    initialPaths,
    tweakGVal);
  bool res = ecbs.runSearch();
  timer.stop();
  std::cout << "Elapsed Time: " << timer.elapsedSeconds() << std::endl;
  if (statFile.size() > 0) {
    std::ofstream stream(statFile);
    stream << timer.elapsedSeconds() << std::endl;
  }

  if (res) {
    cout << "From Driver: Path found" << endl;
  }
  else {
    cout << "From Driver: NO Path found" << endl;
  }

  if (res) {
    ecbs.exportJson(outputFile);
  }

  // print some stats
  if (res) {
    size_t maxT = 0;
    double totalDistance = 0;
    for (const auto& e : ecbs.result()) {
      maxT = std::max(maxT, e->size() - 1);
      for (size_t i = 0; i < e->size() - 1; ++i) {
        totalDistance += (searchGraph[(*e)[i+1]].pos - searchGraph[(*e)[i]].pos).norm();
      }
    }
    std::cout << "T: " << maxT << " totalDistance: " << totalDistance << std::endl;
  }
}

