#pragma once

#include <vector>
#include <unordered_set>
#include <set>

#include "libCommon/map.h"
#include "libCommon/searchgraph.h"

class ECBSSearch;
class Egraph {
public:
  Egraph(
    const std::string& fileName);

  // Egraph(
  //   const Egraph& other);

  Egraph();

  ~Egraph();

  bool isEdge(uint32_t n1, uint32_t n2) const;

  void clear();

  void addEdge(uint32_t n1, uint32_t n2);

  std::set< std::pair<uint32_t, uint32_t> >::const_iterator begin() const {
    return m_set.begin();
  }

  std::set< std::pair<uint32_t, uint32_t> >::const_iterator end() const {
    return m_set.end();
  }

private:
  std::set< std::pair<uint32_t, uint32_t> > m_set;
  // EgraphReader* pImpl;
  // std::string m_fileName; // Hack for copy construction...
};

struct Agent
{
public:
  Agent(
    const std::string& name,
    const std::string& type,
    uint32_t start,
    uint32_t goal)
    : name(name)
    , type(type)
    , start(start)
    , goal(goal)
  {
  }

  std::string name;
  std::string type;
  uint32_t start;
  uint32_t goal;
};


class ECBS {
public:
  ECBS(
    const searchGraph_t* searchGraph,
    const std::vector<Agent>& agents,
    double e_w,
    double e_f,
    const std::vector< std::vector<uint32_t> >& initialPaths,
    bool tweak_g_val = false);

  ~ECBS();

  bool runSearch();

  // agents paths (each entry [ag][t] contains a pathEntry struct)
  const std::vector< const std::vector<uint32_t>* >& result() const;

  void exportJson(const std::string& fileName) const;

private:
  const searchGraph_t* m_searchGraph;
  const std::vector<Agent> m_agents;
  ECBSSearch* pImpl;
};
