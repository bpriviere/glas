#include "ecbs.h"

#include "ecbs_search.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;
#include <boost/tokenizer.hpp>


Egraph::Egraph(
    const std::string& fileName)
  : m_set()
{
  std::string line;
  std::ifstream myfile(fileName.c_str());
  // int num_of_vertices = 0;
  if (myfile.is_open()) {
    getline(myfile, line);
    boost::tokenizer<> tok(line);
    boost::tokenizer<>::iterator beg = tok.begin();
    beg++; beg++;  // skip the prefix of "p edge"
    beg++;
    int num_of_edges = atoi((*beg).c_str());
    for (int i = 0; i < num_of_edges; i++) {
      getline(myfile, line);
      boost::tokenizer<> edge_tok(line);
      boost::tokenizer<>::iterator e_beg = edge_tok.begin();
      e_beg++;
      int from_v = atoi((*e_beg).c_str());
      e_beg++;
      int to_v = atoi((*e_beg).c_str());
      addEdge(from_v, to_v);
    }
    myfile.close();
  }
}

Egraph::Egraph()
  : m_set()
{
}

// Egraph::Egraph(
//   const Egraph& other)
// {
//   m_fileName = other.m_fileName;
//   if (m_fileName.size() > 0) {
//     pImpl = new EgraphReader(m_fileName);
//   } else {
//     pImpl = new EgraphReader();
//   }
// }

// Egraph::Egraph()
//   : m_fileName()
// {
//   pImpl = new EgraphReader();
// }

Egraph::~Egraph()
{
  // delete pImpl;
}

bool Egraph::isEdge(uint32_t n1, uint32_t n2) const
{
  return m_set.find(std::make_pair(n1, n2)) != m_set.end();
}

void Egraph::clear()
{
  m_set.clear();
}

void Egraph::addEdge(uint32_t n1, uint32_t n2)
{
  m_set.insert(std::make_pair(n1, n2));
}

/////////////////////////////////////////////////////////////////////////////

ECBS::ECBS(
    const searchGraph_t* searchGraph,
    const std::vector<Agent>& agents,
    double e_w,
    double e_f,
    const std::vector< std::vector<uint32_t> >& initialPaths,
    bool tweak_g_val)
  : m_searchGraph(searchGraph)
  , m_agents(agents)
  , pImpl(nullptr)
{
  pImpl = new ECBSSearch(searchGraph, agents, e_w, e_f, initialPaths, tweak_g_val);
}

ECBS::~ECBS()
{
  delete pImpl;
}


bool ECBS::runSearch()
{
  return pImpl->runECBSSearch();
}

const std::vector< const std::vector<uint32_t>* >& ECBS::result() const
{
  return pImpl->paths();
}

void ECBS::exportJson(const std::string& fileName) const
{
  // write output file
  using namespace pt;

  ptree pt;
  ptree agents;
  for (size_t ag = 0; ag < pImpl->paths().size(); ag++) {
    ptree agent;
    agent.put("name", m_agents[ag].name);
    agent.put("type", m_agents[ag].type);
    agent.put("group", ag);
    ptree path;
    size_t t = 0;
    for (auto& entry : *(pImpl->paths()[ag])) {
      ptree pathentry;
      pathentry.put("locationId", entry);
      pathentry.put("name", (*m_searchGraph)[entry].name);
      pathentry.put("t", t++);
      const auto& pos = (*m_searchGraph)[entry].pos;
      pathentry.put("x", pos.x());
      pathentry.put("y", pos.y());
      pathentry.put("z", pos.z());

      path.push_back(std::make_pair("", pathentry));
    }
    agent.add_child("path", path);
    agents.push_back(std::make_pair("", agent));
  }
  pt.add_child("agents", agents);
  write_json(fileName, pt);
}
