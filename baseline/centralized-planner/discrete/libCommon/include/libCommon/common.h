#pragma once

#include <stdint.h>
#include <map>
#include <vector>

#include <Eigen/Core>

typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> position_t;

struct hashPos
{
  size_t operator()(const position_t& pos) const
  {
    std::hash<float> h;
    return h(pos.x()) ^ h(pos.y()) ^ h(pos.z());
  }
};

struct cmpPos
{
  bool operator()(const position_t& a, const position_t& b) const
  {
    return a.isApprox(b, 1e-6);
  }
};

struct RobotProperties
{
  uint32_t group;
  double max_v;     // m/s
  bool holonomic;
};

struct Goal
{
  struct singleGoal {
    uint32_t locationId;
  };

  std::map<uint32_t, std::vector<singleGoal> > goalByGroup;
};

// Current state of all robots
struct State
{
  struct robotState {
    position_t position;
    double theta;
  };

  std::vector<robotState> robots;
};
