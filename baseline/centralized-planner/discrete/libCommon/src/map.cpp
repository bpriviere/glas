#include <limits>

#include "libCommon/map.h"

// Boost property tree
// #include <boost/property_tree/ptree.hpp>
// #include <boost/property_tree/json_parser.hpp>

// using namespace boost;

// #include "HelperPropertyTree.hpp"

#include "yaml-cpp/yaml.h"


Map::Map(
  const std::string& fileName)
  : m_obstacles()
  , m_speedLimits()
  , m_dimx(0)
  , m_dimy(0)
  , m_dimz(0)
  , m_dirty(false)
{
#if 0
  // read scene file
  namespace pt = boost::property_tree;

  // Create empty property tree object
  pt::ptree tree;

  // Parse the JSON into the property tree.
  pt::read_json(fileName, tree);

  std::vector<size_t> dimensions = asVector<size_t>(tree, "dimensions");
  m_dimx = dimensions[0];
  m_dimy = dimensions[1];
  m_dimz = dimensions[2];

  // read obstacles
  for (auto& obstacle : tree.get_child("obstacles")) {

      // read position
      std::vector<size_t> obstaclePos = asVector<size_t>(obstacle.second, "");
      position_t pos(obstaclePos[0], obstaclePos[1], obstaclePos[2]);
      uint32_t idx = positionToIdx(pos);
      m_obstacles.insert(idx);
  }
#endif

  YAML::Node map = YAML::LoadFile(fileName);
  const auto& dimensions = map["dimensions"];
  m_dimx = dimensions[0].as<double>();
  m_dimy = dimensions[1].as<double>();
  m_dimz = dimensions[2].as<double>();

  // read obstacles
  for (const auto& obstacle : map["obstacles"]) {
    // read position
    position_t pos(
      obstacle[0].as<double>(),
      obstacle[1].as<double>(),
      obstacle[2].as<double>());
      uint32_t idx = positionToIdx(pos);
      m_obstacles.insert(idx);
  }
}

uint32_t Map::positionToIdx(
  const position_t& position) const
{
  return round(position.x()) + m_dimx * round(position.y()) + m_dimx * m_dimy * round(position.z());
}

position_t Map::idxToPosition(
  uint32_t idx) const
{
  position_t pos;
  pos(0) = idx % m_dimx;
  idx /= m_dimx;
  pos(1) = idx % m_dimy;
  idx /= m_dimy;
  pos(2) = idx;

  return pos;
}

bool Map::isObstacle(
  uint32_t idx) const
{
  auto pos = idxToPosition(idx);
  return pos.x() >= m_dimx
      || pos.y() >= m_dimy
      || pos.z() >= m_dimz
      || m_obstacles.find(idx) != m_obstacles.end();
}

void Map::setObstacle(
  uint32_t idx,
  bool obstacle)
{
  if (obstacle && !isObstacle(idx)) {
    m_obstacles.insert(idx);
    m_dirty = true;
  } else if (isObstacle(idx)) {
    m_obstacles.erase(idx);
    m_dirty = true;
  }
}

float Map::speedLimit(
  uint32_t idx) const
{
  const auto iter = m_speedLimits.find(idx);
  if (iter != m_speedLimits.end()) {
    return iter->second;
  } else {
    return std::numeric_limits<float>::max();
  }
}

void Map::setSpeedLimit(
  uint32_t idx,
  float limit)
{
  m_speedLimits[idx] = limit;
}

void Map::clearSpeedLimit(
  uint32_t idx)
{
  m_speedLimits.erase(idx);
}
