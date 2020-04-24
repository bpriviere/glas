#include "libCommon/AgentsLoaderYaml.h"

#include "yaml-cpp/yaml.h"

AgentsLoaderYaml::AgentsLoaderYaml(
  const std::string& file)
  : m_initialLocations()
  , m_goalLocations()
{
  YAML::Node config = YAML::LoadFile(file);
  for (const auto& agent : config["agents"]) {
    const auto& name = agent["name"];
    m_names.push_back(name.as<std::string>());
    const auto& type = agent["type"];
    m_types.push_back(type.as<std::string>());
    const auto& start = agent["start"];
    const auto& goal = agent["goal"];
    m_initialLocations.push_back(position_t(
      start[0].as<double>(),
      start[1].as<double>(),
      start[2].as<double>()));
    m_goalLocations.push_back(position_t(
      goal[0].as<double>(),
      goal[1].as<double>(),
      goal[2].as<double>()));
  }
}
