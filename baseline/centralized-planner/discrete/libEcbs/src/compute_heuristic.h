#ifndef COMPUTEHEURISTIC_H
#define COMPUTEHEURISTIC_H

#include <unordered_map>
#include "ecbs.h"

class ComputeHeuristic {
public:
  ComputeHeuristic(
    const searchGraph_t* searchGraph,
    uint32_t agent,
    uint32_t goal_location,
    double highwayWeight);

  ~ComputeHeuristic();

  double getHValue(
    uint32_t location) const;

 private:
  std::unordered_map<uint32_t, double> m_hvalues;
};

#endif
