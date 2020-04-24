#pragma once

#include <stdint.h>
#include <string>
#include <unordered_set>
#include <unordered_map>

#include "common.h"

// Static map of the environment; supports discrete and continous mapping
class Map
{
public:
  Map(
    const std::string& fileName);

  uint32_t positionToIdx(
    const position_t& position) const;

  position_t idxToPosition(
    uint32_t idx) const;

  bool isObstacle(
    uint32_t idx) const;

  void setObstacle(
    uint32_t idx,
    bool obstacle);

  float speedLimit(
    uint32_t idx) const;

  void setSpeedLimit(
    uint32_t idx,
    float limit);

  void clearSpeedLimit(
    uint32_t idx);

  size_t dimX() const {
    return m_dimx;
  }

  size_t dimY() const {
    return m_dimy;
  }

  size_t dimZ() const {
    return m_dimz;
  }

  bool isDirty() const {
    return m_dirty;
  }

  void clearDirty() {
    m_dirty = false;
  }

private:
  std::unordered_set<uint32_t> m_obstacles;
  std::unordered_map<uint32_t, float> m_speedLimits;
  size_t m_dimx;
  size_t m_dimy;
  size_t m_dimz;
  bool m_dirty;
};
