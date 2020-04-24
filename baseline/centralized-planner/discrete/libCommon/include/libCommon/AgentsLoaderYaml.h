#pragma once

#include <string>
#include <vector>
#include <utility>

#include "common.h"

class AgentsLoaderYaml {
public:
  AgentsLoaderYaml(
    const std::string& file);

  size_t numAgents() const {
    return m_initialLocations.size();
  }

  const std::vector<position_t>& initialLocations() const
  {
    return m_initialLocations;
  }

  const std::vector<position_t>& goalLocations() const
  {
    return m_goalLocations;
  }

  const std::vector<std::string>& name() const {
    return m_names;
  }

  const std::vector<std::string>& types() const {
    return m_types;
  }

private:

 // public:
  std::vector<std::string> m_names;
  std::vector<std::string> m_types;
  std::vector<position_t> m_initialLocations;
  std::vector<position_t> m_goalLocations;
};
